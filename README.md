# Neural ODE and Deep Euler method test code

This repository contains codes for testing and comparing two neural network based numerical integration techniques. One of them is the Neural ODE method [1] for replacing the original equation with a neural network model. The other one is the Deep Euler method [2] or HyperEuler [3], which uses a neural network to approximate and correct the error of the Euler method.

[1] [Chen et al. Neural Ordinary Differential Equations](https://proceedings.neurips.cc/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf)  
[2] [Shen et al.Deep Euler method: solving ODEs by approximating the local truncation error of the Euler method](https://arxiv.org/abs/2003.09573)  
[3] [Poli et al. Hypersolvers: Toward Fast Continuous-Depth Models](https://proceedings.neurips.cc/paper/2020/file/f1686b4badcf28d33ed632036c7ab0b8-Paper.pdf)

## Running the codes

Steps to run the Neural ODE codes:
1. Run `train.py` with adequate arguments to train a Neural ODE model
1. Use `figures.ipynb` notebook to simulate, plot and evaluate the performance

Steps to run the Deep Euler codes:
1. Use the `data_gen.ipynb` notebook to generate training data
1. Run `dem_train.py` with adequate command line arguments to train a neural network model for use in DEM
1. Use `figures.ipynb` notebook to simulate, plot and evaluate the performance

## Necessary packages

numpy, pytorch, scikit-learn, torchdiffeq, matplotlib, h5py
