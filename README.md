# Text_classification_with_CNN_RNN

This is a repository for text classification with CNN and RNN. The code is based on the paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) and [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://arxiv.org/abs/1605.05101).
This is for the course NLP at the University of Illinois at Urbana-Champaign.

Install the following packages:
```zsh
conda create --name nlp_tutorials  python=3.8
conda activate nlp_tutorials
# the following command works for cuda 11.5 and cuda 11.3
# The PyTorch install,  pytorch-1.10.2,  supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


#   DOES not work ==> cuda version 11.6, for NVIDIA-SMI 510.85.02, Driver Version: 510.85.02 for RTX 3060s

#NVIDIA GeForce RTX 3060 Laptop GPU with CUDA capability sm_86 is not compatible with PyTorch 1.10.2 installation.
# PyTorch 1.12.1 supports CUDA capabilities sm_86 for NVIDIA GeForce RTX 3060 Laptop GPU with cudatoolkit 11.6.0 

# As of 10 Oct 2022, the above command does not work. Use the following instead.
# NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7
# this can be installed using "NVIDIA driver metapackage from nvidia-driver 515 (propriety, tested)" from "Software and Updates/ additional drivers" in Ubuntu 20.04
# You need to have cuda 11.7, but still use the following command to install pytorch 1.12.1
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install torchtext tabulate
pip install torchdata
```

To convert jupyter notebook to python script, run the following command:
```zsh
 jupyter nbconvert --to python hwk2.ipynb 
```

```zsh
conda create --name nlp_cuda116_python3_9 python=3.9   # for cuda 11.6, pytorch 1.12.1
conda activate nlp_cuda116_python3_9 
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda116
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
conda install matplotlib pandas

conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
conda install -c conda-forge jupyterlab
conda install -c conda-forge nb_conda_kernels
conda install -c conda-forge jupyter_contrib_nbextensions

```

To convert to hw2.py, run the following command:
```zsh
jupyter nbconvert --to python hwk2.ipynb  
```



