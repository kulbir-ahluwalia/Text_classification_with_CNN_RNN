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

#NVIDIA GeForce RTX 3060 Laptop GPU with CUDA capability sm_86 is not compatible with the current PyTorch installation.

# As of 10 Oct, the above command does not work. Use the following instead.
#   this is because, the default cuda version is 11.6,  
#   for NVIDIA-SMI 510.85.02, Driver Version: 510.85.02 for RTX 3060
# 
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install torchtext tabulate
pip install torchdata
```

To convert jupyter notebook to python script, run the following command:
```zsh
 jupyter nbconvert --to python hwk2.ipynb 
```

