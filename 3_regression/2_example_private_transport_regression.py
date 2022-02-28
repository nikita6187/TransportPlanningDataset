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
with bz2.BZ2File("1_data_regression.pbz2", 'rb') as handle:
    data_dic = pickle.load(handle)


training_loader = DataLoader(data_dic["training"], batch_size=32, shuffle=True)
validation_loader = DataLoader(data_dic["validation"], batch_size=1, shuffle=False)
validation_extras = data_dic["validation_extras"]
test_loader = DataLoader(data_dic["test"], batch_size=1, shuffle=False)
test_extras = data_dic["test_extras"]
output_scaling = data_dic["output_scaling_prt"] if TRANSPORT_MODE == 0 else data_dic["output_scaling_put"]

num_input_features = data_dic["training"][0].num_node_features
num_output_features = 1


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
        self.lin_prt_2 = torch.nn.Linear(hidden_channels, num_output_features)

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

criterion1 = torch.nn.MSELoss()
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.005)

# Training
running_loss = 0.0
min_pred_val = 10.0  # Skip [0, 10) veh/h due to class imbalance, and as we have good predictor for this
min_pred_val_processed = torch.tensor(min_pred_val) * 1.0 / output_scaling
print("Started training")

for epoch in range(300):  # Training loop - would take a couple of hours - for exploring try setting lower value here

    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs_prt = gcn(data.to(device))

        # Apply mask, skip bucket [0,10) veh/h
        mask_low_vals = data.y[:, TRANSPORT_MODE] > min_pred_val_processed
        final_mask = torch.logical_and(mask_low_vals, data.mask.to(torch.bool))

        # Apply mask
        outputs_prt = torch.squeeze(outputs_prt[final_mask, :])
        y = data.y[:, TRANSPORT_MODE][final_mask]

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
gcn.eval()


for idx, data in zip(range(len(validation_loader)), validation_loader):  # <------- For test set insert test_loader here
    with torch.no_grad():
        pred = gcn(data.to(device))

        pred = pred.cpu()
        data = data.cpu()
        y = data.y[:, TRANSPORT_MODE]

        data.mask = data.mask.to(torch.bool)

        # Apply post processing
        pred_masked = pred[data.mask]
        y_masked = y[data.mask]

        pred_masked = pred_masked * output_scaling
        y_masked = y_masked * output_scaling

        # Mask low values out
        y_low_mask = y_masked >= min_pred_val
        y_masked = y_masked[y_low_mask]
        pred_masked = pred_masked[y_low_mask]

        predicted_outputs_values_weighted.extend(pred_masked.numpy()[:, 0])
        true_outputs_prt_values.extend(y_masked.numpy())


# Calculate and report scores
true_outputs_prt_values = np.asarray(true_outputs_prt_values)
predicted_outputs_values_weighted = np.asarray(predicted_outputs_values_weighted)
rec_mae_weighted = mean_absolute_error(true_outputs_prt_values, predicted_outputs_values_weighted)
rec_r2_weighted = r2_score(true_outputs_prt_values, predicted_outputs_values_weighted)

print("Results")
print("MAE >= 10 veh/h: " + str(rec_mae_weighted))
print("R^2 >= 10 veh/h: " + str(rec_r2_weighted))



