# `Dandage_and_Landry_2019`

Source code for
## [**Paralog dependency indirectly affects the robustness of human cells**](https://doi.org/10.15252/msb.20198871).
Rohan Dandage and Christian R Landry
Institute Of Biology Integrative Et Des Systemes, Universite Laval, Quebec, Canada.
Corresponding author email: christian.landry{at}bio.ulaval.ca
Mol Syst Biol (2019)15:e8871: https://doi.org/10.15252/msb.20198871

## Requirements
1. python 3
2. Anaconda package distributor

## Usage

### Installing dependencies

Required python packages can be installed from the [environment.yml](./environment.yml) file.

    conda env create -f environment.yml

### Installing the package

    git clone https://github.com/rraadd88/human_paralogs.git;cd human_paralogs;pip install -e .
    
### Contents

| filename                                                            | description                                     |
|---------------------------------------------------------------------|-------------------------------------------------|
| [functions.py](./human_paralogs/functions.py)                       | codes for the analysis                          |
| [plots.py](./human_paralogs/plots.py)                               | codes for generating plots                      |
| [figures.ipynb](./human_paralogs/figures.ipynb)                     | codes for generating figures                    |
| [global_vars.py](./human_paralogs/global_vars.py)                   | global variables of the analysis                |
| [cfg.yml](./human_paralogs/cfg.yml)                                 | yaml file with file paths                       |

