# Sift: Channel-Wise Partial Historical Embedding for High Efficiency Distributed Graph Neural Network Training with Accuracy Guarantee

The source code of the Sift framework for high efficiency distributed graph neural network training with accuracy guarantee.
![image](https://github.com/xzwbsz/Sift/assets/44642002/ddd609dc-f0fd-4f22-acfe-1fe2736b697b)

## Basic Requirements

dgl                           1.0.2+cu113 \\
numba                         0.57.0 \\
numpy                         1.24.2 \\
numpydoc                      1.5.0 \\
ogb                           1.3.6 \\
outdated                      0.2.2 \\
packaging                     23.0 \\
PaGraph                       0.1 \\
pandas                        2.0.0 \\
Pillow                        9.5.0 \\
PyYAML                        6.0 \\
scikit-learn                  1.2.2 \\
scipy                         1.10.1 \\
torch                         1.10.1+cu111 \\
torch-cluster                 1.5.9 \\
torch-geometric               2.0.0 \\
torch-scatter                 2.0.9 \\
torch-sparse                  0.6.12 \\
torch-spline-conv             1.2.1 \\
torchaudio                    0.10.1+cu111 \\
torchvision                   0.11.2+cu111 \\
torchviz                      0.0.2 \\

## Autorun Script

```c
bash autorun.sh 
```
<br>
The script depends on fyJu_withSawtooth.py or base_withoutSawtooth.py
Users can change parameter in autorun.sh to test combination under different parallelisms.

## Using Different GNN model
We provide a full function python file 'fyJu_withSawtooth.py', channel-wise replacement without Sawtooth Rearrangement file 'base_withoutSawtooth.py' and other normal benchmark in this project.

Users can replace 'gcn.py' in autorun.sh with any file in this project such as 'fyJu_withSawtooth.py' to test different implementations.



## Acknowledgement
The project is developed based on [Sancus](https://github.com/chenzhao/light-dist-gnn), [GNNAutoScale](https://github.com/rusty1s/pyg_autoscale) and DIGEST for distributed historical embedding mechanism.
