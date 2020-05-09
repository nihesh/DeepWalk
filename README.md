GraphNodeClassification
=======================

This is the Python-3.6 re-implementation of the paper *DeepWalk: Online Learning of Social Representations*. 

Overview
--------

There are two different implementations within this repository. The deep walk re-implementation is located within deepwalk folder. Another implementation that uses negative sampling instead of hierarchical softmax is located in src. We run our experiments on the blog catalog dataset.

Dependencies
------------

* torch-1.2.0
* torchvision-0.4.0
* numpy-1.17.2
* sklearn-0.21.3
* matplotlib-3.0.2
* tqdm
* h5py
* pickle

Deep Walk Re-implementation
---------------------------

The deepwalk re-implementation is included in the deepwalk folder.
To train deepwalk algorithm on BlogCatalog dataset, use
> `$ python3 train.py`

Deepwalk can be trained on any other graph by passing its node list and edge connection files in the same format as BlogCatalog inside the main function.
The final embeddings are stored into `embeddings.pkl` in the same folder.
To change any arguments associated with deepwalk training such as random walk length, modify values present in args.py

> `$ python3 inference.py`
to output f1 scores for the given embeddings file and train-test split ratio. The path to embeddings pkl file and the ratio need to be changed in `inference.py` before proceeding.

Negative sampling implementation
--------------------------------

> `$ python3 train.py`

By default, this command starts training on BlogCatalog dataset using the default parameters set in train.py. The generated embeddings are dumped into `./dump/embeddings.pkl`.

> `$ python3 inference.py`

This script loads the learnt embeddings from `./dump/embeddings.pkl` and trains a linear SVM to report classification accuracy.

> `$ python3 visualise.py`

This script shows the embedding separation between close nodes and far nodes in the graph.

Results
-------

*Re-implementation results*

| Percentage of labelled nodes | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train Macro F1 			   |33.7 | 31.3	| 29.54 |	29.51 |	29.94| 28.22|	30.04|	27|	30.7|  
| Train Micro F1			   |40|	38.7|	36.8|	37.12|	36.6|	35.43|	36.16|	34.83|	36| 
| Test Macro F1				   |17.1|	20.11|	23.55|	23.71|	24.77|	26.68|	26.07|	28.09|	28.15| 
| Test Micro F1				   |28.3|	29.8|	31.9|	31.83|	31.98|	34|	32.94|	35.72|	33.45| 	

*Negative sampling results*

| Percentage of labelled nodes | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train Macro F1 			   | 41.73 | 34.92 | 35.01 | 30.87 | 30.26 | 30.28 | 31.49 | 31.67 | 30.7 |  
| Train Micro F1			   | 46.55 | 43.33 | 42.77 | 41.43 | 41.42 | 35.88 | 36.35 | 36.65 | 36.7 | 
| Test Macro F1				   | 18.27 | 19.43 | 20.9 | 24.85 | 25.13 | 29.33 | 29 | 30.04 | 31.35 | 
| Test Micro F1				   | 29.75 | 31.84 | 32.96 | 33.7 | 34.47 | 34.59 | 34.04 | 34.89 | 36.69 |
