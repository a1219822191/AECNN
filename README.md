#Atomic Environmental Convolutional  Neural Network(AECNN)

This software package uses atomic environmental information to predict material properties

#The package provides two major functions:
1.Train a AECNN model with a customized dataset.
2.Predict material properties of new crystals with a pre-trained AECNN model.


## Usage

python main.py --dataroot=Your Dir contains structure files in xyz format

You can set the number of training, validation, and test data with labels `--train-size`, `--val-size`, and `--test-size`.
