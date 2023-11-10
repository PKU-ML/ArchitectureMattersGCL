# Official code for NeurIPS [Architecture Matters: Uncovering Implicit Mechanisms in Graph Contrastive Learning](https://arxiv.org/abs/2311.02687)


## Introduction

We observe that common phenomena among existing GCL methods that are quite different from VCL methods, including 1) positive samples are not a must for GCL; 2) negative samples are not necessary for graph classification, neither for node classification when adopting specific normalization modules; 3) data augmentations have much less influence on GCL, as simple domain-agnostic augmentations (e.g., Gaussian noise) can also attain fairly good performance. By uncovering how the implicit inductive bias of GNNs works in contrastive learning, we theoretically provide insights into the above intriguing properties of GCL. 

## File Structures

We provide implementations of GCL methods including GRACE, GCA, ProGCL, AutoGCL, BGRL, GraphCL, JOAO, ADGCL, InfoGraph, and a VCL method SimCLR. For running details, please refer to the `README.md` file in the corresponding sub-directory.

``` bash
ContraNorm/
├── README.md
├── GCL/
│   ├── README.md
│   ├── ADGCL/ 
│   │   ├── README.md
│   │   └── ...
│   ├── AutoGCL/
│   │   ├── README.md
│   │   └── ...
│   ├── BGRL/
│   │   ├── README.md
│   │   └── ...
│   ├── GCA/
│   │   ├── README.md
│   │   └── ...
│   ├── GRACE/
│   │   ├── README.md
│   │   └── ...
│   ├── GraphCL/
│   │   ├── README.md
│   │   └── ...
│   ├── InfoGraph/
│   │   ├── README.md
│   │   └── ...
│   ├── ProGCL/
│   │   ├── README.md
│   │   └── ...
│   ├── JOAO/
│   │   ├── README.md
│   │   └── ...
│   └── datasets/
├── VCL/
│   ├── README.md
│   └── ...
└── ...
```

## Citation

If you use our code, please cite

```
@inproceedings{guo2023architecture,
  title={Architecture Matters: Uncovering Implicit Mechanisms in Graph Contrastive Learning},
  author={Xiaojun Guo and Yifei Wang and Zeming Wei and Yisen Wang},
  booktitle={NeurIPS},
  year={2023}
}
```