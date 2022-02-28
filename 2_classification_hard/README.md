# Hard Classification Task


Classification of every link in the graph simultaneously. Note, the links have been transformed to nodes to make it a node classification problem.

In private transportation, the classes depict the following traffic volumes: 
```
bucket_prt = list(range(25, 500, 25))
bucket_prt.extend(list(range(500, 1500, 50)))
bucket_prt.extend(list(range(1500, 2000, 100)))
bucket_prt.extend(list(range(2000, 4501, 500)))
```

For public transportaiton, the classes depict the following traffic volumes: 
```
bucket_put = list(range(25, 300, 25))
bucket_put.extend(list(range(300, 1000, 50)))
bucket_put.extend(list(range(1000, 2000, 100)))
bucket_put.extend(list(range(2000, 4501, 500)))
```

For both, we ignore [0,10) veh/h due to class imbalance, and as we can predict that class very well.
The data is from the simulations done for a morning peak hour.

An example of how to apply a deep GNN to this dataset is provided in ```2_example_private_transport_classification_hard.py```, with comments highlighting how to apply to public transportation as well.







