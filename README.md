# DuoMet
## Introduction
![image](https://github.com/knightmare0227/DuoMet/blob/main/overview.png)

DuoMet is a dual-view representation learning framework that integrates biochemical features, metabolite similarity information, directed graph learning, and hypergraph learning to predict missing metabolic reactions. DuoMet consists of four key components: a biochemical feature learning module that extracts metabolite representations using a model pretrained on large-scale molecular datasets; a similarity-aware feature learning module that captures biochemical similarities among metabolites; a directed graph learning module that models asymmetric substrate-to-product dependencies in metabolic transformations; and a hypergraph learning module that captures higher-order interactions among multiple metabolites participating in the same reaction.

## Environment
All experiments are implemented using Python 3.8, and the required dependencies are listed below.

- torch: 1.12.1
- dgl: 1.1.1+cu116
- torch_geometric: 2.1.0
- numpy: 1.21.5
- pandas: 1.3.5
- scikit-learn: 1.0.2
- tqdm: 4.62.1

## Usage
- Download the pre-trained node embedding file for all metabolites **`metabolite_emb.pkl`** ([download link](https://drive.google.com/file/d/1ASbk0Hyf7E1J3G4BFGQNYNzatOI005dy/view?usp=sharing)) and place it in the main directory.
- python main.py
