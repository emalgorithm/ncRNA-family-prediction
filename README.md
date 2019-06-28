# ncRNA Classification

Code for the paper [ncRNA Classification with Graph Convolutional
Networks](https://arxiv.org/abs/1905.06515) by E. Rossi et. al.

## Citation
To cite our work, please reference our KDD@DLG 2019 paper
```
@inproceedings{rossi2019ncrna,
  title={ncRNA Classification with Graph Convolutional Networks},
  author={Rossi, Emanuele and Monti, Federico and Bronstein, Michael and Li{\`o}, Pietro},
  booktitle={Proceedings of the 1st International Workshop on Deep Learning on Graphs: Methods and Applications (DLG@KDD)},
  year={2019}
}
```

## Requirements
Our code has been tested with python=3.7, but it may be possible for it
to work with lower versions as well (eg. python 3.4-3.6)*[]: 

To install all the required packages it is sufficient to run:

```
pip install -r requirements.txt
```

The setting has been tested for macOS Mojave version 10.14.5, but we expect 
this setup to work with other unix systems.

## Data

- train.fasta: Training dataset. Contains 5670 ncRNA sequences belonging
 to 13 different classes 
- val.fasta: Validation dataset. Contains 650 ncRNA sequences belonging
 to 13 different classes
- test_13_classes.fasta: Full test dataset. Contains 2600 ncRNA 
sequences belonging to 13 different classes
- test_12_classes.fasta: Reduced test dataset. Contains 2400 ncRNA 
sequences belonging to 12 different classes. 
- foldings.pkl: A dictionary containing precomputed mapping from sequence to 
corresponding most likely folding (graph) for all sequences from the above 
datasets.

The data has been taken from 'nRC: non-coding RNA Classifier based on structural features'.
The files train.fasta and val.fasta have been obtained by randomly splitting the 
their original training dataset into two.

## Usage

### Training
It is possible to train the model with the commands:

```
cd src/
python training/train_model.py
```

The hyperparameters of the model can be passed as arguments to the script. 
It is possible to see which hyperparameters can be passed by running (from inside src/):

```
python training/train_model.py --help
```

### Evaluation
The model can be evaluated with the command (from inside src/):

```
python evaluation/evaluate_model.py
```

By default, the model is evaluated on the test_13_classes.fasta test dataset, but
it is possible to change this by passing another test dataset as a script argument.
