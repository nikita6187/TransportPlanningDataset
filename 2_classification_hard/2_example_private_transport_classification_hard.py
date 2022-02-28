import bz2
import pickle
import numpy as np
import torch.nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCN2Conv
import torch.nn.functional as F
import torch.optim
from sklearn.metrics import mean_absolute_error, r2_score


TRANSPORT_MODE = 0   # <------------------------------------------------------------- SET TO 1 FOR PUBLIC TRANSPORTATION


# First load in data
with bz2.BZ2File("1_data_classification_hard.pbz2", 'rb') as handle:
    data_dic = pickle.load(handle)

bucket_prt = list(range(25, 500, 25))
bucket_prt.extend(list(range(500, 1500, 50)))
bucket_prt.extend(list(range(1500, 2000, 100)))
bucket_prt.extend(list(range(2000, 4501, 500)))

bucket_put = list(range(25, 300, 25))
bucket_put.extend(list(range(300, 1000, 50)))
bucket_put.extend(list(range(1000, 2000, 100)))
bucket_put.extend(list(range(2000, 4501, 500)))

buckets = bucket_prt if TRANSPORT_MODE == 0 else bucket_put
buckets.insert(0, 10)

training_loader = DataLoader(data_dic["training"], batch_size=32, shuffle=True)
validation_loader = DataLoader(data_dic["validation"], batch_size=1, shuffle=False)
validation_extras = data_dic["validation_extras"]
test_loader = DataLoader(data_dic["test"], batch_size=1, shuffle=False)
test_extras = data_dic["test_extras"]

num_input_features = data_dic["training"][0].num_node_features
num_output_features_prt = torch.max(data_dic["training"][0].y[:, 0]) + 1
num_output_features_put = torch.max(data_dic["training"][0].y[:, 1]) + 1


# Define GNN - using GCNII as we need larger depths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCN(torch.nn.Module):

    def __init__(self, hidden_channels=128, num_layers=20, alpha=0.1, theta=1.5, dropout=0.5):
        super(GCN, self).__init__()

        self.lin1 = torch.nn.Linear(num_input_features, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha=alpha, theta=theta, layer=layer + 1, shared_weights=True,
                         normalize=True)
            )

        self.dropout = dropout
        self.lin_prt_1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin_prt_2 = torch.nn.Linear(hidden_channels, len(buckets))

    def forward(self, data):

        x_0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = x_1 = self.lin1(x_0)

        for idx in range(0, len(self.convs), 1):

            conv = self.convs[idx]

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, x_1, edge_index, edge_weight)
            x = x.relu()

        prt_x = self.lin_prt_1(x)
        prt_x = prt_x.relu()
        prt_x = self.lin_prt_2(prt_x)

        return F.log_softmax(prt_x, dim=1)


# Run training cycle
gcn = GCN().to(device)
gcn.train()

criterion1 = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001)

# Training
running_loss = 0.0
bucket_to_ignore = 0  # Skip the bucket [0, 10) veh/h due to class imbalance, and as we have good predictor for this
print("Started training")

for epoch in range(300):  # Training loop - would take a couple of hours - for exploring try setting lower value here

    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs_prt = gcn(data.to(device))

        # Apply mask, skip bucket [0,10) veh/h
        mask_low_vals = torch.ne(data.y[:, TRANSPORT_MODE], bucket_to_ignore)
        final_mask = torch.logical_and(mask_low_vals, data.mask.to(torch.bool))
        outputs_prt = outputs_prt[final_mask, :]  # Masking out nodes so we only predict for edges
        targets_prt = data.y[:, TRANSPORT_MODE]
        y = torch.reshape(targets_prt, shape=(-1,))[final_mask]

        loss_prt = criterion1(outputs_prt, y)
        loss = loss_prt
        loss.backward()
        optimizer.step()

        # get statistics
        running_loss += loss.item()

    running_loss = running_loss/len(training_loader)
    print("Epoch: " + str(epoch) + " running_loss: " + str(running_loss))


# Get validation results
predicted_outputs_values_weighted = []
true_outputs_prt_values = []

# Pre-process buckets
buckets_average = []
for idx in range(len(buckets)):
    upper_bound = buckets[idx]
    lower_bound = 0 if idx == 0 else buckets[idx - 1]
    buckets_average.append((upper_bound + lower_bound) / 2.0)
buckets_average = np.asarray(buckets_average)


for idx, data in zip(range(len(validation_loader)), validation_loader):  # <------- For test set insert test_loader here
    with torch.no_grad():
        # Get prediction and targets
        prediction = gcn(data.to(device)).cpu()
        data = data.cpu()
        y = data.y[:, TRANSPORT_MODE]
        data_extras = validation_extras[idx]    # <--------------------------------------- For test set use test_extras here
        y_cpu = y.numpy().astype(int)

        # Get original target values, hacky
        for node_idx in range(y_cpu.shape[0]):
            if node_idx in data_extras["original_nodes_to_edges"]:
                # Ignore buckets [0, 10) veh/h
                if y_cpu[node_idx] == bucket_to_ignore:
                    continue

                u, v = data_extras["original_nodes_to_edges"][node_idx]
                edge_attr_name = "link_prt_volume" if TRANSPORT_MODE == 0 else "link_put_volume_without_walk"
                true_outputs_prt_values.append(
                    data_extras[edge_attr_name][(u, v)])

        # get weighted average predictions, hacky
        out = prediction.numpy()
        rows_to_select = y_cpu != bucket_to_ignore
        out = out[rows_to_select, :]
        output_distribution = np.exp(out)
        buckets_average_extra_dim = np.expand_dims(buckets_average, axis=0)
        output_weighted = np.multiply(output_distribution, buckets_average_extra_dim)
        output_weighted = np.sum(output_weighted, axis=1, keepdims=False)
        predicted_outputs_values_weighted.extend(output_weighted.tolist())


# Calculate and report scores
true_outputs_prt_values = np.asarray(true_outputs_prt_values)
predicted_outputs_values_weighted = np.asarray(predicted_outputs_values_weighted)
rec_mae_weighted = mean_absolute_error(true_outputs_prt_values, predicted_outputs_values_weighted)
rec_r2_weighted = r2_score(true_outputs_prt_values, predicted_outputs_values_weighted)

print("Results")
print("MAE >= 10 veh/h: " + str(rec_mae_weighted))
print("R^2 >= 10 veh/h: " + str(rec_r2_weighted))



