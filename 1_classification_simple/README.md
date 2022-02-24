# Simple Classification Task


Simple classification of every link in the graph simultaneously. Note, the links have been transformed to nodes to make it a node classification problem.

In private transportation, the classes depict the following traffic volumes: 
- 1 - [0, 10) veh/h,  
- 2 - [10, 500) veh/h,  
- 3 - >= 500 veh/h

For public transportaiton, the classes depict the following traffic volumes: 
- 1 - [0, 10) pers/h,  
- 2 - [10, 300) pers/h,  
- 3 - >= 300 pers/h

The data is from the simulations done for a morning peak hour.

An example of how to apply a deep GNN to this dataset is provided in ```2_example_private_transport.py```, with comments highlighting how to apply to public transportation as well.



