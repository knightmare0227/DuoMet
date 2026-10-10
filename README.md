# DuoMet
## Introduction
![image](https://github.com/Xudong-Liang/Multi-HGNN/blob/main/overview.png)

DuoMet is a dual-view representation learning framework that integrates biochemical features, metabolite similarity information, directed graph learning, and hypergraph learning to predict missing metabolic reactions. DuoMet consists of four key components: a biochemical feature learning module that extracts metabolite representations using a model pretrained on large-scale molecular datasets; a similarity-aware feature learning module that captures biochemical similarities among metabolites; a directed graph learning module that models asymmetric substrate-to-product dependencies in metabolic transformations; and a hypergraph learning module that captures higher-order interactions among multiple metabolites participating in the same reaction.

## Environment
We conduct our experiments with python 3.8. Here are the requirements

- torch: 1.13.1
- dgl: 1.1.1+cu116
- torch_geometric: 0.9.3
- numpy: 1.24.4
- pandas: 1.5.3
- scikit-learn: 1.3.0
- tqdm: 4.65.0

## Usage
- Download pre-trained node embedding files for all metabolites **`metabolite_emb.pkl`** ([download link](https://drive.google.com/file/d/1l3tNxv8pLRtyi-iZDGGnGAG7El6e-sTj/view?usp=sharing)) and place it in the main folder.
- python main.py
