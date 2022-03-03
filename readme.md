## Using Deep Learning to Annotate the Protein Universe

Residual Network for Pfam Protein Sequence Annotation

Model Inspired by "Using Deep Learning to Annotate the Protein Universe"

Paper: https://www.biorxiv.org/content/10.1101/626507v2

Trained Model: https://console.cloud.google.com/storage/browser/dbtx-storage/Deeplearning/saved_model/?project=dbtx-pipeline

Data Source: https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam/random_split

### Training

The model was trained using Pytorch Resnet18 with 1 channel and 64 filters from torch vision models on the pfam seed sequences (1086741 Sequences) with 17931 Families in Pfam 32.0 Release. The model was trained with minibatch 100 for 5 epochs, where each epoch switches between the train splits in the data

Training Accuracy: 98.932 % 

### Testing 
The model was then saved and tested on 126171 sequences from Pfam (Full), where the sequences have not been seen at all during training. 

Testing Accuracy: 95.407 %

### Make Predictions:

    # Make Inference on an entire Fasta File 
    python3 inference.py -m fasta -i <Path to Protein Fasta> -mpath ../saved_models/resnet_pfam_final_8.mdl
    # Make Inference from STDIN
    python3 inference.py -m string -i AAQFVAEHGDQVCPAKWTPGAETIVPSL -mpath ../saved_models/resnet_pfam_final_8.mdl
-------------------------
# Google Dataset Description
## Problem description 
This directory contains data to train a model to predict the function of protein domains, based
on the PFam dataset.

Domains are functional sub-parts of proteins; much like images in ImageNet are pre segmented to 
contain exactly one object class, this data is presegmented to contain exactly and only one
domain.

The purpose of the dataset is to repose the PFam seed dataset as a multiclass classification 
machine learning task.
 
The task is: given the amino acid sequence of the protein domain, predict which class it belongs
to. There are about 1 million training examples, and 18,000 output classes.

## Data structure
This data is more completely described by the publication "Can Deep Learning
Classify the Protein Universe", Bileschi et al.

### Data split and layout
The approach used to partition the data into training/dev/testing folds is a random split.

- Training data should be used to train your models.
- Dev (development) data should be used in a close validation loop (maybe
  for hyperparameter tuning or model validation).
- Test data should be reserved for much less frequent evaluations - this
  helps avoid overfitting on your test data, as it should only be used
  infrequently.

### File content
Each fold (train, dev, test) has a number of files in it. Each of those files
contains csv on each line, which has the following fields:

```
sequence: HWLQMRDSMNTYNNMVNRCFATCIRSFQEKKVNAEEMDCTKRCVTKFVGYSQRVALRFAE
family_accession: PF02953.15
sequence_name: C5K6N5_PERM5/28-87
aligned_sequence: ....HWLQMRDSMNTYNNMVNRCFATCI...........RS.F....QEKKVNAEE.....MDCT....KRCVTKFVGYSQRVALRFAE
family_id: zf-Tim10_DDP
```

Description of fields:
- sequence: These are usually the input features to your model. Amino acid sequence for this domain.
  There are 20 very common amino acids (frequency > 1,000,000), and 4 amino acids that are quite 
  uncommon: X, U, B, O, Z.
- family_accession: These are usually the labels for your model. Accession number in form PFxxxxx.y 
  (Pfam), where xxxxx is the family accession, and y is the version number. 
  Some values of y are greater than ten, and so 'y' has two digits.
- family_id: One word name for family.
- sequence_name: Sequence name, in the form "$uniprot_accession_id/$start_index-$end_index".
- aligned_sequence: Contains a single sequence from the multiple sequence alignment (with the rest of the members of 
  the family in seed, with gaps retained.

Generally, the `family_accession` field is the label, and the sequence
(or aligned sequence) is the training feature.

This sequence corresponds to a _domain_, not a full protein.

The contents of these fields is the same as to the data provided in Stockholm
format by PFam at
ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam32.0/Pfam-A.seed.gz


[1] Eddy, Sean R. "Accelerated profile HMM searches." 
    PLoS computational biology 7.10 (2011): e1002195.

## License

Creative Commons Legal Code

CC0 1.0 Universal

    CREATIVE COMMONS CORPORATION IS NOT A LAW FIRM AND DOES NOT PROVIDE
    LEGAL SERVICES. DISTRIBUTION OF THIS DOCUMENT DOES NOT CREATE AN
    ATTORNEY-CLIENT RELATIONSHIP. CREATIVE COMMONS PROVIDES THIS
    INFORMATION ON AN "AS-IS" BASIS. CREATIVE COMMONS MAKES NO WARRANTIES
    REGARDING THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS
    PROVIDED HEREUNDER, AND DISCLAIMS LIABILITY FOR DAMAGES RESULTING FROM
    THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS PROVIDED
    HEREUNDER.

Statement of Purpose

The laws of most jurisdictions throughout the world automatically confer
exclusive Copyright and Related Rights (defined below) upon the creator
and subsequent owner(s) (each and all, an "owner") of an original work of
authorship and/or a database (each, a "Work").

Certain owners wish to permanently relinquish those rights to a Work for
the purpose of contributing to a commons of creative, cultural and
scientific works ("Commons") that the public can reliably and without fear
of later claims of infringement build upon, modify, incorporate in other
works, reuse and redistribute as freely as possible in any form whatsoever
and for any purposes, including without limitation commercial purposes.
These owners may contribute to the Commons to promote the ideal of a free
culture and the further production of creative, cultural and scientific
works, or to gain reputation or greater distribution for their Work in
part through the use and efforts of others.
