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
 and it is called ClinTox. It contains 1478 drugs molecules labelled in 2 classes: 
 toxic and safe.
 
 ## Getting Started
 Since we are going to use packages from both pip and conda you should have 
 [anaconda](https://www.anaconda.com/distribution/?gclid=Cj0KCQiAvJXxBRCeARIsAMSkApqg-qkK5wu2lEGCutGt3Oy0j2GT21HsFtmPyD6Il6VhOVKbPnNM_y8aAu3qEALw_wcB) 
 installed. Than in order to run the notebook do the following steps:
 
 - Create a new environment with the needed packages
 
 ``conda env create -f environment.yml``

 - Activate the environment
 
 ``conda activate atol-env``
 
 Now you should be able to run the notebook! Enjoy it :) 