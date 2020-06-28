![Python 3.6](https://img.shields.io/badge/Python-3.6-green.svg)
![PyTorch 1.1](https://img.shields.io/badge/PyTorch-1.1-blue.svg)
# Structured Domain Adaptation (SDA)

The code for the NeurIPS-2020 submission "Structured Domain Adaptation with Online Relation Regularization for Unsupervised Person Re-ID".

**Please note that**
+ It serves as the *supplementary material* for the anonymous review process.
+ We provide *full testing code* for evaluating our models proposed in the paper.
+ We provide *core training code* `sda_model.py` for reference, and the others are partially hidden for the knowledge protection.
+ The *full* repository containing all training code will be available upon the paper published.


## Prepare Datasets

Download [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [MSMT17](https://arxiv.org/abs/1711.08565) in the directory like
```shell
data/
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
└── msmt17
    └── MSMT17_V1
```

## Test
All the models proposed in our paper could be downloaded by the [link](https://drive.google.com/open?id=1Q2cE0bMrpoyy-g1LR3b-3XclGM1q0dJf), and you could evaluate the model by running the following script
```shell
python test_reid.py --dataset ${DATASET} --resume /path/to/the/checkpoint/
```
where `${DATASET}` indicates the target-domain dataset, e.g. `market1501/dukemtmc/msmt17`.
