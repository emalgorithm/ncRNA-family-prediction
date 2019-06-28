# ncRNA Classification

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
