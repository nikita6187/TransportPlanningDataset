# Regression Task


Regression on every link in the graph simultaneously. Note, the links have been transformed to nodes to make it a node classification problem.

For preprocessing, we divide the original traffic value by 250.

For both PrT and PuT, we ignore [0,10) veh/h due to class imbalance, and as we can predict that class very well.
The data is from the simulations done for a morning peak hour.

An example of how to apply a deep GNN to this dataset is provided in ```2_example_private_transport_regression.py```, with comments highlighting how to apply to public transportation as well.









