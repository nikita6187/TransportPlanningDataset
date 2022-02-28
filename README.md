# Strategic Transport Planning Dataset

A graph based strategic transport planning dataset, aimed at creating the next generation of deep graph neural networks in transfer learning situations. Based on simulation results of the Four Step Model in PTV Visum.

Details of the work as well as results can be found in the thesis ["Development of a Deep Learning Surrogate for the Four-Step Transportation Model"](https://mediatum.ub.tum.de/doc/1638691/dwz10x0l0w38xdklv9zkrprqs.pdf).


<img src="https://raw.githubusercontent.com/nikita68/TransportPlanningDataset/main/1_classification_simple/q1_output_example_11_prediction.PNG" alt="drawing" width="650"/>

_Example of prediction of a city's congestion level, using only socioeconomic and network information as input_

## State of the Art

| **Dataset**                               | **Model** | **Metric** | **Test Performance** | **Publication**                                                                       |
|-------------------------------------------|-----------|------------|----------------------|---------------------------------------------------------------------------------------|
| Classification Simple - Private Transport | GCNII     | Average F1 | 0.87                 | [Makarov, 2021](https://mediatum.ub.tum.de/doc/1638691/dwz10x0l0w38xdklv9zkrprqs.pdf) |
| Classification Simple - Public Transport  | GCNII     | Average F1 | 0.78                 | [Makarov, 2021](https://mediatum.ub.tum.de/doc/1638691/dwz10x0l0w38xdklv9zkrprqs.pdf) |
| Classification Hard - Private Transport   | GCNII     | MAE >=10   | 118.5                | [Makarov, 2021](https://mediatum.ub.tum.de/doc/1638691/dwz10x0l0w38xdklv9zkrprqs.pdf) |
| Classification Hard - Public Transport    | GCNII     | MAE >=10   | 131.8                | [Makarov, 2021](https://mediatum.ub.tum.de/doc/1638691/dwz10x0l0w38xdklv9zkrprqs.pdf) |
| Regression - Private Transport            | GCNII     | MAE >=10   | 135.4*               | [Makarov, 2021](https://mediatum.ub.tum.de/doc/1638691/dwz10x0l0w38xdklv9zkrprqs.pdf) |
| Regression - Public Transport             | GCNII     | MAE >=10   | 141.2*               | [Makarov, 2021](https://mediatum.ub.tum.de/doc/1638691/dwz10x0l0w38xdklv9zkrprqs.pdf) |

**If you reach a better performance, please issue a pull request with the new values!**

_* Regression model used only augmented dataset for training data_ 

"MAE >=10" is the mean absolute error, but only applied to test samples where the target has at least 10 units/h.
This is done as the lower values can be predicted accurately and are less interesting for the domain. An implementation can
be found in the examples and details in the thesis above.


## Dataset Overview

| Dataset                 | Example | Training Samples | Validation Samples | Test Samples | Task                        | Primary Metric | Graph Sizes            | Private Transport | Public Transport | Input & Output Transformed |
|-------------------------|------------------ |------------------|--------------------|--------------|-----------------------------|----------------|------------------------|-------------------|------------------|----------------------------|
| [1_classification_simple](https://github.com/nikita68/TransportPlanningDataset/tree/main/1_classification_simple) | [Example](https://github.com/nikita68/TransportPlanningDataset/blob/main/1_classification_simple/2_example_private_transport.py) | 6398             | 1600*              | 2000*        | Classification - 3 classes  | Average F1     | 15 - 80 original nodes | Yes               | Yes              | Yes                        |
| [2_classification_hard](https://github.com/nikita68/TransportPlanningDataset/tree/main/2_classification_hard)   | [Example](https://github.com/nikita68/TransportPlanningDataset/blob/main/2_classification_hard/2_example_private_transport_classification_hard.py) | 16393            | 1600*              | 2000*        | Classification - 51 classes | MAE >=10       | 15 - 80 original nodes | Yes               | Yes              | Yes                        |
| [3_regression](https://github.com/nikita68/TransportPlanningDataset/tree/main/3_regression)            | [Example](https://github.com/nikita68/TransportPlanningDataset/blob/main/3_regression/2_example_private_transport_regression.py) | 16393            | 1600*              | 2000*        | Regression                  | MAE >=10       | 15 - 80 original nodes | Yes               | Yes              | Yes                        |

_* Identical validation and test datasets_

All of the data is pickled and compressed with pbz2, with the datasets being ready to be used in PyTorch Geometric. Please see the examples on how to use the data.

## Problem Setting

The aim of strategic transport planning is to do long term predictions of a given city, based purely on the underlying transport network and socioecenomic data. The socioeconomic data of every household is grouped into so called zones. To solve the problem the model needs to both understand how socioeconomic data creates demand as well as it is applied to the supply side of the network. In practise, this is usually solved by the [4 Step Model](https://www.transitwiki.org/TransitWiki/index.php/Four-step_travel_model), however it has a number of downsides including requiring a large amount of manual work to calibrate and slow prediction speed.

This problem is a great task for deep graph neural networks. However, there is no large public dataset available, so we propose to generate data using in a surrogate model setup, using the 4 Step Model as the baseline. The problem is a transfer learning for GNNs. Within this project, the aim is to create a proof of concept, with all generated cities having between 15 and 80 nodes, and 3-10 zones. Details can be found in [thesis](https://mediatum.ub.tum.de/doc/1638691/dwz10x0l0w38xdklv9zkrprqs.pdf).


## Data Generation

**Augmented Dataset**
- Extract random subnetworks from processed OpenStreetMaps and procedural generation for socioeconomic data
- 6398 training samples, 1600 validation samples, 2000 test samples

**Synthetic Dataset**
- Procedural generation for both network and socioeconomic data
- +9995 training samples


The dataset ```1_classification_simple``` uses exclusively the augmented dataset. Both ```2_classification_hard``` and ```3_regression``` use the augmented dataset for training, validation and test sets, but also add the synthetic dataset for additional training samples. All problems use identical validation and test datasets, with the targets transformed to the respective task.


## Data Transformation

- The datasets presented have all input features standardized and filtered as needed
- As most common GNNs can only do simulatenous predictions for nodes, all edges are transformed into nodes, thus requiring masking during training
    - Irrelevant node/edge features are then set to 0

All details can be found in the thesis above.


## Open Challenges for GNNs

If you need inspiration for what to focus on to improve GNNs, here are some open challenges with details in the thesis above:
- Recurrent GNNs - dynamic depth based on size of input graph
- Overcoming the [bottleneck of GNNs](https://arxiv.org/abs/2006.05205) - seen severely on this dataset
- Seeing whether we can automatically correct nodes propagating errors


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



## License & Acknowledgements

- Parts of the original data © OpenStreetMap contributors
- A big thank you goes to [Carlos Llorca Garcia](https://www.mos.ed.tum.de/en/tb/team/cl/) and TUM's MSM institute for providing the pre-processed OSM data
- This dataset (TransportPlanningDataset) is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/. Any rights in individual contents of the database are licensed under the Database Contents License: http://opendatacommons.org/licenses/dbcl/1.0/


