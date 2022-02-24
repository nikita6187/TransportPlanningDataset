# Strategic Transport Planning Dataset

A graph based strategic transport planning dataset, aimed at creating the next generation of deep graph neural networks in transfer learning situations. Based on simulation results of the Four Step Model in PTV Visum.

Details of the work as well as results can be found in the thesis ["Development of a Deep Learning Surrogate for the Four-Step Transportation Model"](https://mediatum.ub.tum.de/doc/1638691/dwz10x0l0w38xdklv9zkrprqs.pdf).




## Problem Setting

The aim of strategic transport planning is to do long term predictions of a given city, based purely on the underlying transport network and socioecenomic data. The socioeconomic data of every household is grouped into so called zones. To solve the problem the model needs to both understand how socioeconomic data creates demand as well as it is applied to the supply side of the network. In practise, this is usually solved by the [4 Step Model](https://www.transitwiki.org/TransitWiki/index.php/Four-step_travel_model), however it has a number of downsides including requiring a large amount of manual work to calibrate and slow prediction speed.

This problem is a great task for deep graph neural networks. However, there is no large public dataset available, so we propose to generate data using in a surrogate model setup, using the 4 Step Model as the baseline. The problem is a transfer learning for GNNs. Within this project, the aim is to create a proof of concept, with all generated cities having between 15 and 80 nodes, and 3-10 zones. Details can be found in [thesis](https://mediatum.ub.tum.de/doc/1638691/dwz10x0l0w38xdklv9zkrprqs.pdf).


## Data Generation

**Augmented Dataset**
- Extract random subnetworks from processed OpenStreetMaps and procedural generation for socioeconomic data

**Synthetic Dataset**
- Procedural generation for both network and socioeconomic data


The dataset ```1_classification_simple``` uses exclusively the augmented dataset. Both ```2_classification_hard``` and ```3_regression``` use the augmented dataset for training, validation and test sets, but also add the synthetic dataset for additional training samples. All problems use identical validation and test datasets, with the targets transformed to the respective task.


## Data Transformation

TODO


## Issues & Dataset Requests

If you find any issues with the data or want to get a specific version of the data, please raise an issue over at the top.



## Citing

If you create any new work based on this dataset, and it is only using the data:

```
@misc{makarov2021,
    author = {Makarov, Nikita and Narayanan, Santhanakrishnan and Antoniou, Constantinos},
    institution = {Transportation Systems Engineering},
    school = {Technical University of Munich},
    title = {Development of a Deep Learning Surrogate for the Four-Step Transportation Model},
    year = 2021,
    url = {https://github.com/nikita68/TransportPlanningDataset}
}

```

If you also refer to the previous baselines, please also cite:

```
@mastersthesis{makarov2021,
    author = {Makarov, Nikita},
    institution = {Transportation Systems Engineering},
    school = {Technical University of Munich},
    title = {Development of a Deep Learning Surrogate for the Four-Step Transportation Model},
    year = 2021,
    url = {https://mediatum.ub.tum.de/doc/1638691/dwz10x0l0w38xdklv9zkrprqs.pdf}
}
```



## License

- Parts of the original data © OpenStreetMap contributors
- This dataset (TransportPlanningDataset) is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/. Any rights in individual contents of the database are licensed under the Database Contents License: http://opendatacommons.org/licenses/dbcl/1.0/


