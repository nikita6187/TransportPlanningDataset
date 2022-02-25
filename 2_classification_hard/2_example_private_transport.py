import bz2
import pickle
import numpy as np
import torch.nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCN2Conv
import torch.nn.functional as F
import torch.optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score


TRANSPORT_MODE = 0   # <------------------------------------------------------------- SET TO 1 FOR PUBLIC TRANSPORTATION


# First load in data
with bz2.BZ2File("1_data_classification_simple.pbz2", 'rb') as handle:
    data_dic = pickle.load(handle)

training_loader = DataLoader(data_dic["training"], batch_size=32, shuffle=True)
validation_loader = DataLoader(data_dic["validation"], batch_size=1, shuffle=False)
test_loader = DataLoader(data_dic["test"], batch_size=1, shuffle=False)

num_input_features = data_dic["training"][0].num_node_features
num_output_features_prt = torch.max(data_dic["training"][0].y[:, 0]) + 1
num_output_features_put = torch.max(data_dic["training"][0].y[:, 1]) + 1


# Define GNN - using GCNII as we need larger depths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCN(torch.nn.Module):

    def __init__(self, hidden_channels=256, num_layers=40, alpha=0.1, theta=1.5, dropout=0.5):
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
        self.lin_prt_2 = torch.nn.Linear(hidden_channels, num_output_features_prt if TRANSPORT_MODE == 0 else num_output_features_put)

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
print("Started training")

for epoch in range(300):  # Training loop - would take a couple of hours - for exploring try setting lower value here

    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs_prt = gcn(data.to(device))

        # Apply mask
        outputs_prt = outputs_prt[data.mask.to(torch.bool), :]  # Masking out nodes so we only predict for edges
        targets_prt = data.y[:, TRANSPORT_MODE]
        y = torch.reshape(targets_prt, shape=(-1,))[data.mask.to(torch.bool)]

        loss_prt = criterion1(outputs_prt, y)
        loss = loss_prt
        loss.backward()
        optimizer.step()

        # get statistics
        running_loss += loss.item()

    running_loss = running_loss/len(training_loader)
    print("Epoch: " + str(epoch) + " running_loss: " + str(running_loss))


# Get validation results
predicted_outputs_prt = []
true_outputs_prt = []

for idx, data in zip(range(len(validation_loader)), validation_loader):  # <-------- For testset insert test_loader here

    # Get prediction and targets
    prediction = gcn(data.to(device))
    _, prediction = prediction.max(dim=1)
    prediction = prediction.cpu()
    data = data.cpu()
    y = data.y[:, TRANSPORT_MODE]

    # Get masks
    data.mask = data.mask.to(torch.bool)
    pred_masked = prediction[data.mask]
    y_masked = y[data.mask]

    # Record data
    predicted_outputs_prt.append(pred_masked)
    true_outputs_prt.append(y_masked)


# Calculate and report scores
true_outputs_prt = np.concatenate(true_outputs_prt, axis=0)
predicted_outputs_prt = np.concatenate(predicted_outputs_prt, axis=0)
print("Confusion Matrix:")
print(confusion_matrix(true_outputs_prt, predicted_outputs_prt))
print("Validation scores PrT:")
print(classification_report(true_outputs_prt, predicted_outputs_prt))

