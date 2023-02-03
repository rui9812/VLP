# VLP
To Copy Rather Than Memorize: A Vertical Learning Paradigm for Knowledge Graph Completion

## Requirements
- pytorch == 1.9.0
- dgl-cu111
- graph_tool

## Data
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

## Models
- [x] RotatE-VLP
- [x] ComplEx-VLP
- [x] DistMult-VLP

## Usage
All training commands are listed in [best_config.sh](./best_config.sh). 
For example, you can run the following commands to train RotatE-VLP on WN18RR and FB15k-237 datasets.
```
# WN18RR
bash run.sh train RotatE wn18rr 0 0 512 1024 500 4.0 0.5 0.00005 80000 8 1 0.5 5.0 8 0.5 1.1 10 30000 -de

# FB15k-237
bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 11.0 1.0 0.0005 100000 16 -1 0.5 0.5 5 3.0 1.7 13 40000 -de
```

## Acknowledgement
We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). Thanks for their contributions.
