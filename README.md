# Axon Projection

A code that analyses long-range axons provided as input, and classifies them based on the brain regions they project to.


## Installation

First, create a virtual environment with at least python 3.10, and activate it:
```bash
python -m venv venvAP
source venvAP/bin/activate
```

The preferred way is to install directly from pypi:
```bash
pip install axon-projection
```

Otherwise, you can still clone this repository and install the package:
```bash
git clone https://github.com/BlueBrain/axon-projection.git
cd axon-projection
pip install .
```

## Running

```bash
python run.py <config_file>
```
or
```bash
axon-projection -c <config_file>
```
## Workflow

The workflow can either be run entirely, from clustering to the sampling, by running the `run.py` script. Each step can also be run separately, by running modules individually.
Parameters for each step can be configured in a `.cfg` file, called at execution, *i.e.*:
```bash
python axonal_projections.py config.cfg
```

|<p align="center"><img src="./docs/APWorkflow.png" alt="Axonal projection workflow" width="100%" height="auto"></p>|
|:---:|
| *Overview of the workflow. LRA : Long-range axon.* |

These steps are executed in the following order:
- **axonal_projections.py**: creates a table that contains source region and number of terminals or axonal length in each terminal region, for every provided morphology. Hierarchy level of source and target regions can be controlled in the configuration file (the higher the level, the deeper into regions).
<!-- - **check_atlas.py** (optional): compares source regions found for the morphologies in the provided atlas, with source regions found elsewhere, typically from another atlas or manually assigned. Also checks the discrepancies between targeted regions for each morphology (*n.b.*: morphologies files tested should be the same and at the same disk location). -->
- **classify_axons.py**: runs the clustering on the axonal projections table. Each morphology is grouped by source region, and feature vectors are defined by the number of terminals in each target region. Clustering is unsupervised, and done by Gaussian Mixture Models (GMMs). The number of mixture components (*i.e.* number of clusters) for each source is selected to minimize the Bayesian Information Criterion, which balances the likelihood of the dataset by the number of parameters used in the model.
The output of this step, is the creation of clusters for each source region, defined by :
    - a probability to belong to this cluster;
    - the mean number of terminals or axonal length in each target region;
    - the variances of this feature (terminals or lengths);

  and the assignment to each cluster for every morphology in the dataset.
- **visualize_connections.py** (optional): for each cluster, creates a graph of connectivity to other regions. Connectivity strengths are also shown, computed as $s = \frac{N_r}{N_T}$, with $N_r$ is the total number of terminals in the target region in the entire cluster, divided by $N_T$, the total number of terminals of all the axons in this cluster.

|<p align="center"><img src="./docs/graph_example.png" alt="Example graph" width="60%" height="auto"></p>|
|:---:|
| *Orange nodes are for source region, purple for target regions, and blue for intermediary hierarchy to traverse (*i.e.*: DG-mo is in DG, which is in HIP, etc...).* |


- **separate_tufts.py**: clusters and saves the tufts of each morphology by region, with their topological barcodes. Also computes how each tuft is representative of its group, defined by GMM cluster and target region, by comparing the difference of the tuft with all the others tufts of its group, based on a set of morphometrics (defined in the configuration file). This representativity score ranges from 0 (not representative) to 1 (representative). Finally, this step also computes *trunk_morphometric* morphometrics on the trunks of these morphologies (data needed for axon-synthesis).
- **sample_axon.py**: uses the previously defined GMMs to sample an axon from a specified source region. This draws a cluster assignment, and a number of terminals or axon length in each target region.
<!-- Appropriate tufts are then selected, based on this number of terminals and the tufts' representativity score. The output is a tuft tuple, which, among others, contains the tuft topological barcode, which can be used for topological synthesis. -->


## Examples
The example folder contains some files to run an example usage of the code.

### Clustering of axons
First, brain atlas files (`brain_atlas.zip`) need to be downloaded from https://doi.org/10.5281/zenodo.13790069 and uncompressed (for instance with `unzip brain_atlas.zip`) in the example/data folder.

The example/config_clustering_example.cfg configuration file provides the parameters for each step of the workflow. The workflow can be run by executing the script example/run_clustering_example.sh (with the venv activated, source venvAP/bin/activate), which basically runs the complete workflow:

```bash
python ../axon_projection/run.py config_clustering_example.cfg
```

The example should run in a few minutes, depending on the computing configuration (4 minutes with 20 processes). The output is generated in the example/out_clustering_example folder.

### Axon synthesis example
This second example shows how to use the output of the projections clustering from the previous example, as input for the axon synthesis algorithm presented in https://doi.org/10.1101/2024.10.16.618695. Namely, the target locations of the axons, number and selection of tufts is provided by the projection clustering code.

To run the synthesis example, ensure to first run the clustering example.

Then, simply run the run_synthesis_example.sh script with the virtual environment activated. The [example/config_synthesis_example.cfg](example/config_synthesis_example.cfg) configuration file contains the parameters used in the axon synthesis algorithm.

This example synthesizes axons for the set of axon-less morphologies in the folder [example/data/morphologies_for_synthesis](example/data/morphologies_for_synthesis), using the data extracted from axons in the folder [example/data/morphologies](example/data/morphologies).

This example has an initial computation of brain regions masks that takes around 15-20 minutes (and creates the output `boundary.nrrd` in the data directory), and the synthesis runs typically in 5 minutes, without leveraging parallelism.

To use parallelism, one can install the axon-synthesis package with the mpi option
```bash
pip install axon-synthesis[mpi]
```

## Citation

To cite this repository, please cite the accompanying research article:

Petkantchin, R., Berchet, A., Peng, H., Markram, H., Kanari, L., 2024. Generating brain-wide connectome using synthetic axonal morphologies. https://doi.org/10.1101/2024.10.04.616605


## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research
center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH
Board of the Swiss Federal Institutes of Technology.

For license and authors, see `LICENSE.txt` and `AUTHORS.md` respectively.

Copyright © 2023-2024 Blue Brain Project/EPFL
