#Atomic Environmental Convolutional  Neural Network(AECNN)

This software package uses atomic environmental information to predict material properties

#The package provides two major functions:
1.Train a AECNN model with a customized dataset.
2.Predict material properties of new crystals with a pre-trained AECNN model.

## Table of Contents

- [Prerequisties](#prerequistes)
- [Usage](#usage)
    -[Install libwacsf](#install-libwacsf)
    -[Structure enconding and define a customized dataset]
    -[Train a AECNN model]
    -[Predict material properties with a pre-trained AECNN model]
- [Authors]
- [License]


##  Prerequisites

This package requires:
- [PyTorch]
- [ase]
- [h5py]
- [pymatgen]

## Usage
### Step1 : Installing  libwacsf
Entering Dir ${wACSF}, running command
```bash
python setup.py install
```
to install the package in your machine ,if you use system's python, you need root

### Step2 : Structure encoding,and generating training set file struc.h5 by script /tools/AECNN_data_deal.py
```bash
python AECNN_data_deal.py --dataroot=Your Dir contains structure files in cif format
```
You'll get a directory containing.h5 files,
and Place the file containing the filename and target properties in this directory

### Step3 :Training AECNN
in Dir ${AECNN}
```
python main.py --dataroot=./h5_data --train-size=80 --val-size=10 --test-size=10 > log &
```
You can set the number of training, validation, and test data with labels `--train-size`, `--val-size`, and `--test-size`.
