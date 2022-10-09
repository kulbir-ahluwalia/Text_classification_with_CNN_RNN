# Text_classification_with_CNN_RNN

This is a repository for text classification with CNN and RNN. The code is based on the paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) and [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://arxiv.org/abs/1605.05101).
This is for the course NLP at the University of Illinois at Urbana-Champaign.

Install the following packages:
```zsh
conda create --name nlp_tutorials  python=3.8
conda activate nlp_tutorials
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install torchtext tabulate
pip install torchdata
```

To convert jupyter notebook to python script, run the following command:
```zsh
 jupyter nbconvert --to python hwk2.ipynb 
```

