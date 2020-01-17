# Graph classification with Automatic Topologically-Oriented Learning

The analysis in this repository is based on the [paper](https://arxiv.org/pdf/1909.13472.pdf) ATOL: Automatic Topologically-Oriented Learning and the corresponding [source code](https://github.com/martinroyer/atol) by Martin Royer (copyright INRIA). We also adapted the Perslay [codebase](https://github.com/MathieuCarriere/perslay) to match the API from giotto-learn.


## Requisites
  * giotto-learn
  * scikit-learn
  * joblib
  * numpy
  * GUDHI
 
 ## Notebook
 The notebook contains an example of usage for the ATOL layer. It is applyied 
 to a graph classification problem and it reach good results, comparable to 
 state of the art algorithms. 
 
 ### Data
 The dataset we used in the notebook is taken from moleculenet.ai 
 and it is called ClinTox. It contains 1498 drugs molecules labelled in 2 classes: 
 toxic and safe.
