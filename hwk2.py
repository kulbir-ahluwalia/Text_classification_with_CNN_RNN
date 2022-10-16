#!/usr/bin/env python
# coding: utf-8

# # CS 447 Homework 2 $-$ Text Classification with Neural Networks
# In this homework, you will build machine learning models to detect the sentiment of movie reviews using the IMDb movie reviews dataset. Specifically, you will implement classifiers based on Convolutional Neural Networks (CNN's) and Recurrent Neural Networks (RNN's).
# 
# In addition to the Pytorch tutorial we have provided on Coursera, we highly recommend that you take a look at the PyTorch tutorials before starting this assignment:
# <ul>
# <li><a href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html">https://pytorch.org/tutorials/beginner/pytorch_with_examples.html</a>
# <li><a href="https://pytorch.org/tutorials/beginner/data_loading_tutorial.html">https://pytorch.org/tutorials/beginner/data_loading_tutorial.html</a>
# <li><a href="https://github.com/yunjey/pytorch-tutorial">https://github.com/yunjey/pytorch-tutorial</a>
# </ul>
# 
# <font color='green'>While you work, we suggest that you keep your hardware accelerator set to "CPU" (the default for Colab). However, when you have finished debugging and are ready to train your models, you should select "GPU" as your runtime type. This will speed up the training of your models. You can find this by going to <TT>Runtime > Change Runtime Type</TT> and select "GPU" from the dropdown menu.</font>
# 
# As usual, you should not import any other libraries.

# In[1]:


### DO NOT EDIT ###

import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    print('Using device:', DEVICE)


# # Step 1: Download the Data
# First we will download the dataset using [torchtext](https://torchtext.readthedocs.io/en/latest/index.html), which is a package that supports NLP for PyTorch. 
# 
# Unfortunately, you have to install the <TT>torchdata</TT> package on the Colab machine in order to access the data. To do this, run the cell below (you may need to click the "Restart Runtime" button when it finishes). You will have to do this every time you return to work on the homework.
# 
# 

# In[2]:


get_ipython().system('pip install torchdata')


# The following cell will get you `train_data` and `test_data`. It also does some basic tokenization.
# 
# *   To access the list of textual tokens for the *i*th example, use `train_data[i][1]`
# *   To access the label for the *i*th example, use `train_data[i][0]`

# In[3]:


### DO NOT EDIT ###

import torchtext
import random

def preprocess(review):
    '''
    Simple preprocessing function.
    '''
    res = []
    for x in review.split(' '):
        remove_beg=True if x[0] in {'(', '"', "'"} else False
        remove_end=True if x[-1] in {'.', ',', ';', ':', '?', '!', '"', "'", ')'} else False
        if remove_beg and remove_end: res += [x[0], x[1:-1], x[-1]]
        elif remove_beg: res += [x[0], x[1:]]
        elif remove_end: res += [x[:-1], x[-1]]
        else: res += [x]
    return res

if __name__=='__main__':
    train_data = torchtext.datasets.IMDB(root='.data', split='train')
    train_data = list(train_data)
    train_data = [(x[0], preprocess(x[1])) for x in train_data]
    train_data, test_data = train_data[0:10000] + train_data[12500:12500+10000], train_data[10000:12500] + train_data[12500+10000:], 

    print('Num. Train Examples:', len(train_data))
    print('Num. Test Examples:', len(test_data))


    # for i in range(len(train_data)):
    for i in range(0,10,1):
        print(f"the label for the *i*th example: {train_data[i][0]} ")
        print(f"list of textual tokens for the *i*th example: {train_data[i][1]} ")


    # *   To access the list of textual tokens for the *i*th example, use `train_data[i][1]`
    # *   To access the label for the *i*th example, use `train_data[i][0]`


    print("\nSAMPLE DATA:")
    for x in random.sample(train_data, 5):
        print('Sample text:', x[1])
        print('Sample label:', x[0], '\n')


# # Step 2: Create Dataloader [20 points]
# 
# 
# 

# ## <font color='red'>TODO:</font> Define the Dataset Class [20 Points]
# 
# In the following cell, we will define the <b>dataset</b> class. The dataset contains the tokenized data for your model. You need to implement the following functions: 
# 
# *   <b>` build_dictionary(self)`:</b>  <b>[10 points]</b> Creates the dictionaries `idx2word` and `word2idx`. You will represent each word in the dataset with a unique index, and keep track of this in these dictionaries. Use the hyperparameter `threshold` to control which words appear in the dictionary: a training word’s frequency should be `>= threshold` to be included in the dictionary.
# 
# * <b>`convert_text(self)`:</b> Converts each review in the dataset to a list of indices, given by your `word2idx` dictionary. You should store this in the `textual_ids` variable, and the function does not return anything. If a word is not present in the  `word2idx` dictionary, you should use the `<UNK>` token for that word. Be sure to append the `<END>` token to the end of each review.
# 
# *   <b>` get_text(self, idx) `:</b> Return the review at `idx` in the dataset as an array of indices corresponding to the words in the review. If the length of the review is less than `max_len`, you should pad the review with the `<PAD>` character up to the length of `max_len`. If the length is greater than `max_len`, then it should only return the first `max_len` words. The return type should be `torch.LongTensor`.
# 
# *   <b>`get_label(self, idx) `</b>: Return the value `1` if the label for `idx` in the dataset is `positive`, and should return `0` if it is `negative`. The return type should be `torch.LongTensor`.
# 
# *  <b> ` __len__(self) `:</b> Return the total number of reviews in the dataset as an `int`.
# 
# *   <b>` __getitem__(self, idx)`:</b> <b>[10 points]</b> Return the (padded) text, and the label. The return type for both these items should be `torch.LongTensor`. You should use the ` get_label(self, idx) ` and ` get_text(self, idx) ` functions here.
# 
# 
# <b>Note:</b> You should convert all words to lower case in your functions.
# 
# <font color='green'><b>Hint:</b> Make sure that you use instance variables such as `self.threshold` throughout your code, rather than the global variable `THRESHOLD` (defined later on). The variable `THRESHOLD` will not be known to the autograder, and the use of it within the class will cause an autograder error.</font>

# In[4]:


PAD = '<PAD>'
END = '<END>'
UNK = '<UNK>'

from torch.utils import data
from collections import defaultdict

class TextDataset(data.Dataset):
    def __init__(self, examples, split, threshold, max_len, idx2word=None, word2idx=None):
        ### DO NOT EDIT ###
        
        self.examples = examples
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.threshold = threshold
        self.max_len = max_len

        #custom
        self.map_token_to_unigram_frequency = {}

        # Dictionaries
        self.idx2word = idx2word
        self.word2idx = word2idx
        if split == 'train':
            self.build_dictionary()
        self.vocab_size = len(self.word2idx)

        self.vocabulary = []
        
        # Convert text to indices
        self.textual_ids = []
        self.convert_text()

    # def default_function_for_default_dict(self):
    #     return
    def build_dictionary(self): 
        '''
        Build the dictionaries idx2word and word2idx. This is only called when split='train', as these
        dictionaries are passed in to the __init__(...) function otherwise. Be sure to use self.threshold
        to control which words are assigned indices in the dictionaries.
        Returns nothing.
        '''
        assert self.split == 'train'
        
        # Don't change this
        self.idx2word = {0:PAD, 1:END, 2: UNK}
        self.word2idx = {PAD:0, END:1, UNK: 2}

        ##### TODO #####
        # Count the frequencies of all words in the training data (self.examples)
        # Assign idx (starting from 3) to all words having word_freq >= self.threshold
        # Make sure you call word.lower() on each word to convert it to lowercase
        # print(" BUILDING DICT")
        # print(f"training dataset is: {self.examples[0]}")

        # ('pos', ['Your', 'life', 'is', 'good', 'when', 'you', 'have', 'money', ',', 'success', 'and', 'health'])

        # map_token_to_unigram_frequency = defaultdict(lambda: " not")

        for build_dict_data in self.examples:           #iterate through preprocessed sentences
            # print(f"data: {data}")
            testset_label, preprocessed_sentence = build_dict_data[0], build_dict_data[1]
            # print(f"testset_label: {testset_label}, preprocessed_sentence:{preprocessed_sentence}")
            for token in preprocessed_sentence:             #iterate through each token
                # print(f"lowercase token is: {token}")
                token = token.lower()

                if token not in self.map_token_to_unigram_frequency.keys():
                    self.map_token_to_unigram_frequency[token] = 0          # add token as key if not present as a key in map_token_to_unigram_frequency
                if token in self.map_token_to_unigram_frequency.keys():
                    self.map_token_to_unigram_frequency[token] += 1          # add token as key if not present as a key in map_token_to_unigram_frequency


        print(f"self.map_token_to_unigram_frequency: {self.map_token_to_unigram_frequency}")

        # Also make sure your vocabulary is deterministic - if you make it multiple times,
        # you should always get the same word to index mapping.
        # Otherwise, you could get an issue where at test time you are basically mapping words to
        # different indexes than it was trained under, causing basically random embeddings
        # and thus 50% accuracy.

        # if has_passed and sorted(list(dataset.idx2word.keys())) != list(range(0, dataset.vocab_size)):
        #     has_passed, message = False, 'dataset.idx2word must have keys ranging from 0 to dataset.vocab_size-1. Keys in your dataset.idx2word: ' + str(sorted(list(dataset.idx2word.keys())))

        items = self.map_token_to_unigram_frequency.items()
        # print(f"items is: {items}")

        # sort in lexicographical order
        sorted_items = (sorted(self.map_token_to_unigram_frequency.items()))
        print(sorted_items)


        # Made the dictionaries such that each word was always mapped to a particular index.
        # Then removed all indexes and renamed them to (0,len(vocab)-1).
        token_indice = 3
        for token, freq in sorted(self.map_token_to_unigram_frequency.items()):
            # print(f"token: {token}, freq: {freq}")
            # self.idx2word[index] = token
            if freq >= self.threshold:
                # self.idx2word[token_indice] = token
                self.word2idx[token] = token_indice
            token_indice += 1

        for index, (token, token_indice) in enumerate(self.word2idx.items()):
            self.idx2word[index] = token
            self.word2idx[token] = index
            print(f"index: {index}, token: {token}, token_indice{token_indice}")

        # for token, token_indice in temp_dict.keys():
        #     if freq >= self.threshold:


        # vocab_size = len(self.word2idx)
        # self.vocabulary = [self.idx2word[i] for i in range(vocab_size)]


        # --- TEST: idx2word and word2idx dictionaries ---
        #     items is: dict_items([('your', 2), ('life', 3), ('is', 3), ('good', 2), ('when', 3), ('you', 3), ('have', 2), ('money', 2), (',', 2), ('success', 2), ('and', 2), ('health', 2), ('bad', 2), ('got', 2), ('not', 2), ('a', 2), ('lot', 2)])


        print(f"self.idx2word: {self.idx2word}")
        print(f"self.word2idx: {self.word2idx}")

        pass
    
    def convert_text(self):
        '''
        Convert each review in the dataset (self.examples) to a list of indices, given by self.word2idx.
        Store this in self.textual_ids; returns nothing.
        '''

        ##### TODO #####
        # Remember to replace a word with the <UNK> token if it does not exist in the word2idx dictionary.
        # Remember to append the <END> token to the end of each review.
        for convert_text_example in self.examples:           #iterate through preprocessed sentences
            # print(f"convert_text_example: {convert_text_example}")
            testset_label, preprocessed_sentence = convert_text_example[0], convert_text_example[1]
            converted_preprocessed_sentence = []
            # print(f"testset_label: {testset_label}, preprocessed_sentence:{preprocessed_sentence}")
            for token_index in range(len(preprocessed_sentence)):             #iterate through each token
                token = preprocessed_sentence[token_index]
                # print(f"IN CONVERT TEXT: token: {token}")
                token = token.lower()
                # print(f"lowercase token is: {token}")
                if token not in self.word2idx.keys():
                    # preprocessed_sentence[token_index] = self.word2idx[UNK]
                    converted_preprocessed_sentence.append(self.word2idx[UNK])
                if token in self.word2idx.keys():
                    # preprocessed_sentence[token_index] = self.word2idx[token]
                    converted_preprocessed_sentence.append(self.word2idx[token])

            converted_preprocessed_sentence.append(self.word2idx[END])

            # print(f"testset_label: {testset_label}, converted preprocessed_sentence:{converted_preprocessed_sentence}")
            # self.textual_ids.append([testset_label,converted_preprocessed_sentence])
            self.textual_ids.append(converted_preprocessed_sentence)
            # print(f"self.textual_ids is: {self.textual_ids}")

        pass

    def get_text(self, idx):
        '''
        Return the review at idx as a long tensor (torch.LongTensor) of integers corresponding to the words in the review.
        You may need to pad as necessary (see above).

        <b>` get_text(self, idx) `:</b> Return the review at `idx` in the dataset as an array of indices corresponding to the words in the review. If the length of the review is less than `max_len`, you should pad the review with the `<PAD>` character up to the length of `max_len`. If the length is greater than `max_len`, then it should only return the first `max_len` words. The return type should be `torch.LongTensor`.
        '''

        ##### TODO #####
        review = self.examples[idx]
        indice_review = self.textual_ids[idx]
        # print(f"indice review is: {indice_review}")

        if len(indice_review) >= self.max_len:
            indice_review_long_tensor = torch.LongTensor(indice_review[0:self.max_len])

        elif len(indice_review) < self.max_len:
            padding = [self.word2idx[PAD]] * (self.max_len - len(indice_review))
            padded_version = indice_review + padding
            # print(f"padding list is: {padding} and padded version: {padded_version}")
            # indice_review_long_tensor = padded_version.type(torch.int64)
            indice_review_long_tensor = torch.LongTensor(padded_version)


        # print('Content of indice_review_long_tensor:', indice_review_long_tensor)
        # print('Shape of indice_review_long_tensor:', indice_review_long_tensor.shape, '\n')
        # print('Type of indice_review_long_tensor:', indice_review_long_tensor.dtype, '\n')



        return (indice_review_long_tensor)
    
    def get_label(self, idx):
        '''
        This function should return the value 1 if the label for idx in the dataset is 'positive', 
        and 0 if it is 'negative'. The return type should be torch.LongTensor.
        '''
        ##### TODO #####
        review_label = self.examples[idx][0]
        # print(f"review label: {review_label}")

        if review_label == 'pos':
            return torch.squeeze(torch.LongTensor([1]))
            # print('Shape of torch.squeeze(torch.Tensor(1)):', torch.squeeze(torch.Tensor(1)).shape, '\n')
            # return torch.squeeze(torch.Tensor([1]))

        elif review_label == 'neg':
            return torch.squeeze(torch.LongTensor([0]))
            # print('Shape of torch.squeeze(torch.Tensor(0)):', torch.squeeze(torch.Tensor(0)).shape, '\n')
            # return torch.squeeze(torch.Tensor([0]))



    def __len__(self):
        '''
        Return the number of reviews (int value) in the dataset
        '''
        ##### TODO #####
        return int(len(self.examples))
    
    def __getitem__(self, idx):
        '''
        Return the review, and label of the review specified by idx.

        *   <b>` __getitem__(self, idx)`:</b> <b>[10 points]</b> Return the (padded) text, and the label. The return type for both these items should be `torch.LongTensor`. You should use the ` get_label(self, idx) ` and ` get_text(self, idx) ` functions here.
        '''
        ##### TODO #####
        # return self.examples[idx][1], self.examples[idx][0]
        return self.get_text(idx), self.get_label(idx)


# ##Sanity Check: Dataset Class
# 
# The code below runs a sanity check for your `Dataset` class. The tests are similar to the hidden ones in Gradescope. However, note that passing the sanity check does <b>not</b> guarantee that you will pass the autograder; it is intended to help you debug.

# In[5]:


### DO NOT EDIT ###

def sanityCheckDataSet():
    #	Read in the sample corpus
    reviews = [('pos', 'Your life is good when you have money, success and health'),
               ('neg', 'Life is bad when you got not a lot')]
    data = [(x[0], preprocess(x[1])) for x in reviews]
    print("Sample dataset:")
    for x in data: print(x)

    thresholds = [1,2,3]
    print('\n--- TEST: idx2word and word2idx dictionaries ---') # max_len does not matter for this test
    correct = [[',', '<END>', '<PAD>', '<UNK>', 'a', 'and', 'bad', 'good', 'got', 'have', 'health', 'is', 'life', 'lot', 'money', 'not', 'success', 'when', 'you', 'your'], ['<END>', '<PAD>', '<UNK>', 'is', 'life', 'when', 'you'], ['<END>', '<PAD>', '<UNK>']]
    for i in range(len(thresholds)):
        dataset = TextDataset(data, 'train', threshold=thresholds[i], max_len=3)

        has_passed, message = True, ''
        if has_passed and (dataset.vocab_size != len(dataset.word2idx) or dataset.vocab_size != len(dataset.idx2word)):
            has_passed, message = False, 'dataset.vocab_size (' + str(dataset.vocab_size) + ') must be the same length as dataset.word2idx (' + str(len(dataset.word2idx)) + ') and dataset.idx2word ('+str(len(dataset.idx2word)) +').'
        if has_passed and (dataset.vocab_size != len(correct[i])):
            has_passed, message = False, 'Your vocab size is incorrect. Expected: ' + str(len(correct[i])) + '\tGot: ' + str(dataset.vocab_size)
        if has_passed and sorted(list(dataset.idx2word.keys())) != list(range(0, dataset.vocab_size)):
            has_passed, message = False, 'dataset.idx2word must have keys ranging from 0 to dataset.vocab_size-1. Keys in your dataset.idx2word: ' + str(sorted(list(dataset.idx2word.keys())))
        if has_passed and sorted(list(dataset.word2idx.keys())) != correct[i]:
            has_passed, message = False, 'Your dataset.word2idx has incorrect keys. Expected: ' + str(correct[i]) + '\tGot: ' + str(sorted(list(dataset.word2idx.keys())))
        if has_passed: # Check that word2idx and idx2word are consistent
            widx = sorted(list(dataset.word2idx.items())) 
            idxw = sorted(list([(v,k) for k,v in dataset.idx2word.items()]))
            if not (len(widx) == len(idxw) and all([widx[q] == idxw[q] for q in range(len(widx))])):
                has_passed, message = False, 'Your dataset.word2idx and dataset.idx2word are not consistent. dataset.idx2word: ' + str(dataset.idx2word) + '\tdataset.word2idx: ' + str(dataset.word2idx)

        status = 'PASSED' if has_passed else 'FAILED'
        print('\tthreshold:', thresholds[i], '\tmax_len:', 3, '\t'+status, '\t'+message)
    
    print('\n--- TEST: len(dataset) ---')
    has_passed = len(dataset) == 2
    if has_passed: print('\tPASSED')
    else: print('\tlen(dataset) is incorrect. Expected: 2\tGot: ' + str(len(dataset)))

    print('\n--- TEST: __getitem__(self, idx) ---')
    max_lens = [3,8,15]
    idxes = [0,1]
    combos = [{'threshold': t, 'max_len': m, 'idx': idx} for t in thresholds for m in max_lens for idx in idxes]
    correct = [(torch.tensor([3, 4, 5]), torch.tensor(1)), (torch.tensor([ 4,  5, 15]), torch.tensor(0)), (torch.tensor([ 3,  4,  5,  6,  7,  8,  9, 10]), torch.tensor(1)), (torch.tensor([ 4,  5, 15,  7,  8, 16, 17, 18]), torch.tensor(0)), (torch.tensor([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  1,  0,  0]), torch.tensor(1)), (torch.tensor([ 4,  5, 15,  7,  8, 16, 17, 18, 19,  1,  0,  0,  0,  0,  0]), torch.tensor(0)), (torch.tensor([2, 3, 4]), torch.tensor(1)), (torch.tensor([3, 4, 2]), torch.tensor(0)), (torch.tensor([2, 3, 4, 2, 5, 6, 2, 2]), torch.tensor(1)), (torch.tensor([3, 4, 2, 5, 6, 2, 2, 2]), torch.tensor(0)), (torch.tensor([2, 3, 4, 2, 5, 6, 2, 2, 2, 2, 2, 2, 1, 0, 0]), torch.tensor(1)), (torch.tensor([3, 4, 2, 5, 6, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0]), torch.tensor(0)), (torch.tensor([2, 2, 2]), torch.tensor(1)), (torch.tensor([2, 2, 2]), torch.tensor(0)), (torch.tensor([2, 2, 2, 2, 2, 2, 2, 2]), torch.tensor(1)), (torch.tensor([2, 2, 2, 2, 2, 2, 2, 2]), torch.tensor(0)), (torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]), torch.tensor(1)), (torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0]), torch.tensor(0))]



    for i in range(len(combos)):
        combo = combos[i]
        dataset = TextDataset(data, 'train', threshold=combo['threshold'], max_len=combo['max_len'])
        returned = dataset.__getitem__(combo['idx'])

        has_passed, message = True, ''
        if has_passed and len(returned) != 2:
            has_passed, message = False, 'dataset.__getitem__(idx) must return 2 things. Got ' + str(len(returned)) +' things instead.'
        if has_passed and (type(returned[0]) != torch.Tensor or type(returned[1]) != torch.Tensor):
            has_passed, message = False, 'Both returns must be of type torch.Tensor. Got: (' + str(type(returned[0])) + ', ' + str(type(returned[1])) + ')'
        if has_passed and (returned[0].shape != correct[i][0].shape):
            has_passed, message = False, 'Shape of first return is incorrect. Expected: ' + str(correct[i][0].shape) + '.\tGot: ' + str(returned[0].shape)
        if has_passed and (returned[1].shape != correct[i][1].shape):
            has_passed, message = False, 'Shape of second return is incorrect. Expected: ' + str(correct[i][1].shape) + '.\tGot: ' + str(returned[1].shape) + '\n\t\tHint: torch.Size([]) means that the tensor should be dimensionless (just a number). Try squeezing your result.'
        if has_passed and (returned[1] != correct[i][1]):
            has_passed, message = False, 'Label (second return) is incorrect. Expected: ' + str(correct[i][1]) + '.\tGot: ' + str(returned[1])
        if has_passed:
            correct_padding_idxes, your_padding_idxes = torch.where(correct[i][0] == 0)[0], torch.where(returned[0] == dataset.word2idx[PAD])[0]
            if not (correct_padding_idxes.shape == your_padding_idxes.shape and torch.all(correct_padding_idxes == your_padding_idxes)):
                has_passed, message = False, 'Padding is not correct. Expected padding indxes: ' + str(correct_padding_idxes) + '.\tYour padding indexes: ' + str(your_padding_idxes)

        status = 'PASSED' if has_passed else 'FAILED'
        print('\tthreshold:', combo['threshold'], '\tmax_len:', combo['max_len'] , '\tidx:', combo['idx'], '\t'+status, '\t'+message)

if __name__ == '__main__':
    sanityCheckDataSet()


# The following cell builds the dataset on the IMDb movie reviews and prints an example:

# In[6]:


### DO NOT EDIT ###

if __name__=='__main__':
    train_dataset = TextDataset(train_data, 'train', threshold=10, max_len=150)
    print('Vocab size:', train_dataset.vocab_size, '\n')

    randidx = random.randint(0, len(train_dataset)-1)
    text, label = train_dataset[randidx]
    print('Example text:')
    print(train_data[randidx][1])
    print(text)
    print('\nExample label:')
    print(train_data[randidx][0])
    print(label)


# # Step 3: Train a Convolutional Neural Network (CNN) [40 points]

# ## <font color='red'>TODO:</font> Define the CNN Model [20 points]
# Here you will define your convolutional neural network for text classification. We provide you with the CNN class, you need to fill in parts of the `__init__(...)` and `forward(...)` functions. Each of these functions is worth 10 points.
# 
# We have provided you with instructions and hints in the comments. In particular, pay attention to the desired shapes; you may find it helpful to print the shape of the tensors as you code. It may also help to keep PyTorch documentation open for the modules & functions you are using, since they describe input and output dimensions.

# In[7]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, out_channels, filter_heights, stride, dropout, num_classes, pad_idx):
        super(CNN, self).__init__()
        
        ##### TODO #####
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # Create an embedding layer (https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
        #   to represent the words in your vocabulary. Make sure to use vocab_size, embed_size, and pad_idx here.
        # print(f" vocab size is: {vocab_size}, embed size is: {embed_size}, filter heights: {filter_heights}")
        # number of embeddings = vocab size??
        # embedding size = embed size? = max len??
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        #self.embedding is learnable

        # print(f"weights of embedding: {self.embedding.weight}")


        # Define multiple Convolution layers (nn.Conv2d) with filter (kernel) size [filter_height, embed_size] based on your 
        #   different filter_heights.
        # Input channels will be 1 and output channels will be out_channels (these many different filters will be trained
        #   for each convolution layer)


        # If you want, you can store a list of modules inside nn.ModuleList.
        self.conv_module_list = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(filter_height, embed_size), stride=stride) for filter_height in filter_heights])
        # print(f"conv module list is: {self.conv_module_list}")

        self.conv_model_outputs = {}

        # conv module list is: ModuleList(
        #                                   (0): Conv2d(1, 32, kernel_size=(3, 32), stride=(1, 1))
        #                                   (1): Conv2d(1, 32, kernel_size=(4, 32), stride=(1, 1))
        #                                   (2): Conv2d(1, 32, kernel_size=(5, 32), stride=(1, 1))
        #                                 )

        # self.maxpool = nn.MaxPool1d(2)

        # Note: even though your conv layers are nn.Conv2d, we are doing a 1d convolution since we are only moving the filter 
        #   in one direction

        # Create a dropout layer (nn.Dropout) using dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Define a linear layer (nn.Linear) that consists of num_classes units
        #   and takes as input the $concatenated output$ for all cnn layers (out_channels * num_of_cnn_layers units) ???
        # print(f"out_channels * len(filter_heights), :{out_channels * len(filter_heights)}")
        self.dense_layer = nn.Linear(out_channels * len(filter_heights), num_classes)
        # another dense layer
        # self.d2 = nn.Linear(128, 10)


    def forward(self, texts):
        """
        texts: LongTensor [batch_size, max_len]
        
        Returns output: Tensor [batch_size, num_classes]

        # Pass these texts to each of your conv layers and compute their output as follows:
        #   Your cnn output will have shape [batch_size, out_channels, *, 1] where * depends on filter_height and stride
        #   Convert to shape [batch_size, out_channels, *] (see torch's squeeze() function)
        #   Apply non-linearity on it (F.relu() is a commonly used one. Feel free to try others)
        #   Take the max value across last dimension to have shape [batch_size, out_channels]
        # Concatenate (torch.cat) outputs from all your cnns [batch_size, (out_channels*num_of_cnn_layers)]
        #
        """
        ##### TODO #####
        # print('Content of texts:', texts)
        # print('Shape of texts:', texts.shape, '\n')
        # print('Type of texts:', texts.dtype, '\n')

        texts = texts.type(torch.int64)
        final_embedding = self.embedding(texts)




        # print('Content of embedding:', final_embedding)
        # print('Shape of embedding:', final_embedding.shape, '\n')
        # print('Type of embedding:', final_embedding.dtype, '\n')
        # Resulting: shape: [batch_size, max_len, embed_size]
        # Shape of embedding: torch.Size([1, 150, 16])


        # Input to conv should have 1 channel. Take a look at torch's unsqueeze() function
        #   Resulting shape: [batch_size, 1, MAX_LEN, embed_size]
        y = torch.unsqueeze(final_embedding, 1)
        # print('Shape of unsqueeze:', y.shape, '\n')  #CORRECT

        # Pass these texts to each of your conv layers and compute their output as follows:
        #   Your cnn output will have shape [batch_size, out_channels, *, 1] where * depends on filter_height and stride
        #   Convert to shape [batch_size, out_channels, *] (see torch's squeeze() function)
        #   Apply non-linearity on it (F.relu() is a commonly used
        #   one. Feel free to try others)

        list_to_be_concatenated = []

        for i, conv_model in enumerate(self.conv_module_list):

            # Shape of x: torch.Size([1, 32, 148, 1])   #148 = * = depends on filter_height and stride
            # Shape of x1_squeezed: torch.Size([1, 32, 148])
            # Shape of x1_squeezed_relu: torch.Size([1, 32, 148])

            # print(f"i: {i}, conv_model: {conv_model}")
            x = conv_model(y)
            # print('Shape of x:', x.shape, '\n')  #

            x1_squeezed = torch.squeeze(x, dim=3)
            # print('Shape of x1_squeezed:', x1_squeezed.shape, '\n')

            x1_squeezed_relu = F.relu(x1_squeezed) #??? #TODO: apply non linearity here??
            # print('Shape of x1_squeezed_relu:', x1_squeezed_relu.shape, '\n')

            #   Take the max value across last dimension to have shape [batch_size, out_channels] ??? #TODO: how and why?
            # torch.nn.functional.max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
            maxpool_values, maxpool_indices = F.max_pool1d(x1_squeezed_relu, x1_squeezed_relu.shape[2], return_indices=True)
            # print(f"maxpool_values is: {maxpool_values}")
            # print(f"maxpool_values.shape: {maxpool_values.shape}")

            # print(f"maxpool_indices is: {maxpool_indices}, maxpool_indices.shape: {maxpool_indices.shape}")

            maxpool_values_squeezed = torch.squeeze(maxpool_values, dim=2)
            # print(f" maxpool_values_squeezed is: {maxpool_values_squeezed}, maxpool_values_squeezed.shape: {maxpool_values_squeezed.shape}")
            list_to_be_concatenated.append(maxpool_values_squeezed)

        ##########
        # i: 0, conv_model: Conv2d(1, 32, kernel_size=(3, 16), stride=(1, 1))
        # i: 1, conv_model: Conv2d(1, 32, kernel_size=(4, 16), stride=(1, 1))
        # i: 2, conv_model: Conv2d(1, 32, kernel_size=(5, 16), stride=(1, 1))
        # shape of tensor: torch.Size([1, 32, 148])
        # shape of tensor: torch.Size([1, 32, 147])
        # shape of tensor: torch.Size([1, 32, 146])
        ##########
        #  Concatenate (torch.cat) outputs from all your cnns [batch_size, (out_channels*num_of_cnn_layers)]
        list_to_be_concatenated_tensor = torch.cat([i for i in list_to_be_concatenated], dim=1)

        # print(f"list_to_be_concatenated_tensor is: {list_to_be_concatenated_tensor}, shape: {list_to_be_concatenated_tensor.shape}")
        # print(f"list_to_be_concatenated_tensor: {list_to_be_concatenated_tensor.shape}")

        #  #print shapes
        # for tensor in list_to_be_concatenated:
        #     print(f"shape of tensor: {tensor.shape}")

        #[batch_size, (out_channels*num_of_cnn_layers)]
        # shape of tensor: torch.Size([20, 32])
        # shape of tensor: torch.Size([20, 32])
        # shape of tensor: torch.Size([20, 32])
        # list_to_be_concatenated_tensor: torch.Size([20, 96])


        # Let's understand what you just did:
        #   Since each cnn is of different filter_height, it will look at different number of words at a time
        #     So, a filter_height of 3 means your cnn looks at 3 words (3-grams) at a time and tries to extract some information from it
        #   Each cnn will learn out_channels number of features from the words it sees at a time
        #   Then you applied a non-linearity and took the max value for all channels
        #     You are essentially trying to find important n-grams from the entire text
        # Everything happens on a batch simultaneously hence you have that additional batch_size as the first dimension


        # flattened_tensor = torch.flatten(list_to_be_concatenated_tensor)
        # print(f"flattened_tensor is: {flattened_tensor}, shape: {flattened_tensor.shape}")

        # Apply dropout
        dropout = self.dropout_layer(list_to_be_concatenated_tensor)
        # print(f"dropout is: {dropout}, shape: {dropout.shape}")

        # Pass your output through the linear layer and return its output 
        #   Resulting shape: [batch_size, num_classes]
        # x = self.dense_layer(self.relu(x))

        linear = (self.dense_layer(dropout))
        # print(f"linear is: {linear}, shape: {linear.shape}, linear dtype: {linear.dtype}")


        # linear = linear.type(torch.int32)

        ##### NOTE: Do not apply a sigmoid or softmax to the final output - done in training method!


        return linear
        # return None


# ##Sanity Check: CNN Model
# 
# The code below runs a sanity check for your `CNN` class. The tests are similar to the hidden ones in Gradescope. However, note that passing the sanity check does <b>not</b> guarantee that you will pass the autograder; it is intended to help you debug.

# In[8]:


### DO NOT EDIT ###

count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

def sanityCheckModel(all_test_params, NN, expected_outputs, init_or_forward, data_loader):
    # print('--- TEST: ' + ('Number of Model Parameters (tests __init__(...))' if init_or_forward=='init' else 'Output shape of forward(...)') + ' ---')
    
    if init_or_forward == "forward":
        # Reading the first batch of data for testing
        for texts_, labels_ in data_loader:
            texts_batch, labels_batch = texts_, labels_
            break

    for tp_idx, (test_params, expected_output) in enumerate(zip(all_test_params, expected_outputs)):       
        if init_or_forward == "forward":
            batch_size = test_params['batch_size']
            texts = texts_batch[:batch_size]

        # Construct the student model
        tps = {k:v for k, v in test_params.items() if k != 'batch_size'}
        stu_nn = NN(**tps)

        if init_or_forward == "forward":
            with torch.no_grad(): 
                stu_out = stu_nn(texts)
            ref_out_shape = expected_output

            has_passed = torch.is_tensor(stu_out)
            if not has_passed: msg = 'Output must be a torch.Tensor; received ' + str(type(stu_out))
            else: 
                has_passed = stu_out.shape == ref_out_shape
                msg = 'Your Output Shape: ' + str(stu_out.shape)
            

            status = 'PASSED' if has_passed else 'FAILED'
            message = '\t' + status + "\t Init Input: " + str({k:v for k,v in tps.items()}) + '\tForward Input Shape: ' + str(texts.shape) + '\tExpected Output Shape: ' + str(ref_out_shape) + '\t' + msg
            print(message)
        else:
            stu_num_params = count_parameters(stu_nn)
            ref_num_params = expected_output
            comparison_result = (stu_num_params == ref_num_params)

            status = 'PASSED' if comparison_result else 'FAILED'
            message = '\t' + status + "\tInput: " + str({k:v for k,v in test_params.items()}) + ('\tExpected Num. Params: ' + str(ref_num_params) + '\tYour Num. Params: '+ str(stu_num_params))
            print(message)

        del stu_nn


if __name__ == '__main__':
    # Test init
    inputs = [{'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 3, 'pad_idx': 0}]
    expected_outputs = [22434, 22531, 22434, 22531, 23874, 23939, 23874, 23939, 41730, 42115, 41730, 42115, 47490, 47747, 47490, 47747, 44578, 44675, 44578, 44675, 47554, 47619, 47554, 47619, 82306, 82691, 82306, 82691, 94210, 94467, 94210, 94467]

    sanityCheckModel(inputs, CNN, expected_outputs, "init", None)
    print()

    # Test forward
    inputs = [{'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 32, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [3, 4, 5], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 1, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 32, 'out_channels': 128, 'filter_heights': [5, 10], 'stride': 3, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 20}]
    expected_outputs = [torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2]), torch.Size([1, 2]), torch.Size([20, 2])]
    sanity_dataset = TextDataset(train_data, 'train', 5, 150)
    sanity_loader = torch.utils.data.DataLoader(sanity_dataset, batch_size=50, shuffle=True, num_workers=2, drop_last=True)

    sanityCheckModel(inputs, CNN, expected_outputs, "forward", sanity_loader)


# In[9]:


# class MyModule(nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
#
#     def forward(self, x):
#         # ModuleList can act as an iterable, or be indexed using ints
#         for i, l in enumerate(self.linears):
#             print(f"i: {i}, l: {l}")
#
#             first_part = self.linears[i // 2](x)
#             second_part =  l(x)
#             x = first_part + second_part
#
#         return x


# In[10]:


# test_object = MyModule()
# test_object.forward(torch.ones(10,10))


# ## Train CNN Model
# 
# First, we initialize the train and test <b>dataloaders</b>. A dataloader is responsible for providing batches of data to your model. Notice how we first instantiate datasets for the train and test data, and that we use the training vocabulary for both.
# 
# You do not need to edit this cell.

# In[11]:


if __name__=='__main__':
    THRESHOLD = 5 # Don't change this
    MAX_LEN = 200 # Don't change this
    BATCH_SIZE = 128 # Feel free to try other batch sizes

    train_dataset = TextDataset(train_data, 'train', THRESHOLD, MAX_LEN)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    test_dataset = TextDataset(test_data, 'test', THRESHOLD, MAX_LEN, train_dataset.idx2word, train_dataset.word2idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


# Now we provide you with a function that takes your model and trains it on the data.
# 
# You do not need to edit this cell. However, you may want to write code to save your model periodically, as Colab connections are not permanent. See the tutorial here if you wish to do this: https://pytorch.org/tutorials/beginner/saving_loading_models.html.

# In[12]:


### DO NOT EDIT ###

from tqdm.notebook import tqdm

def train_model(model, num_epochs, data_loader, optimizer, criterion):
    print('Training Model...')
    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        epoch_acc = 0
        for texts, labels in data_loader:
            texts = texts.to(DEVICE) # shape: [batch_size, MAX_LEN]
            labels = labels.to(DEVICE) # shape: [batch_size]
            # print(f"labels is: {labels}, labels shape: {labels.shape}, labels dtype: {labels.dtype}")

            optimizer.zero_grad()

            output = model(texts)
            # print(f"output is: {output}, output shape: {output.shape}, output dtype: {output.dtype}")

            acc = accuracy(output, labels)
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print('[TRAIN]\t Epoch: {:2d}\t Loss: {:.4f}\t Train Accuracy: {:.2f}%'.format(epoch+1, epoch_loss/len(data_loader), 100*epoch_acc/len(data_loader)))
    print('Model Trained!\n')


# Here are some other helper functions we will need.

# In[13]:


### DO NOT EDIT ###

def accuracy(output, labels):
    """
    Returns accuracy per batch
    output: Tensor [batch_size, n_classes]
    labels: LongTensor [batch_size]
    """
    preds = output.argmax(dim=1) # find predicted class
    correct = (preds == labels).sum().float() # convert into float for division 
    acc = correct / len(labels)
    return acc


# Now you can instantiate your model. We provide you with some recommended hyperparameters; you should be able to get the desired accuracy with these, but feel free to play around with them.

# In[14]:


if __name__=='__main__':
    cnn_model = CNN(vocab_size = train_dataset.vocab_size, # Don't change this
                embed_size = 128, 
                out_channels = 64, 
                filter_heights = [2, 3, 4], 
                stride = 1, 
                dropout = 0.5, 
                num_classes = 2, # Don't change this
                pad_idx = train_dataset.word2idx[PAD]) # Don't change this

    # Put your model on the device (cuda or cpu)
    cnn_model = cnn_model.to(DEVICE)
    
    print('The model has {:,d} trainable parameters'.format(count_parameters(cnn_model)))


# Next, we create the **criterion**, which is our loss function: it is a measure of how well the model matches the empirical distribution of the data. We use cross-entropy loss (https://en.wikipedia.org/wiki/Cross_entropy).
# 
# We also define the **optimizer**, which performs gradient descent. We use the Adam optimizer (https://arxiv.org/pdf/1412.6980.pdf), which has been shown to work well on these types of models.

# In[15]:


import torch.optim as optim

if __name__=='__main__':    
    LEARNING_RATE = 5e-4 # Feel free to try other learning rates

    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Define the optimizer
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)


# Finally, we can train the model. If the model is implemented correctly and you're using the GPU, this cell should take around <b>4 minutes</b> (or less). Feel free to change the number of epochs.

# In[16]:


if __name__=='__main__':
    N_EPOCHS = 15 # Feel free to change this == 20 == best

    # train model for N_EPOCHS epochs
    train_model(cnn_model, N_EPOCHS, train_loader, optimizer, criterion)


# ## Evaluate CNN Model [20 points]
# 
# Now that we have trained a model for text classification, it is time to evaluate it. We have provided you with a function to do this; you do not need to modify anything.
# 
# To pass the autograder for the CNN, you will need to achieve **82% accuracy** on the hidden test set on Gradescope. Note that the Gradescope test set is very similar, and the accuracies between the two datasets should be comparable.

# In[17]:


### DO NOT EDIT ###

import random

def evaluate(model, data_loader, criterion, use_tqdm=False):
    print('Evaluating performance on the test dataset...')
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    print("\nSOME PREDICTIONS FROM THE MODEL:")
    iterator = tqdm(data_loader) if use_tqdm else data_loader
    total = 0
    for texts, labels in iterator:
        bs = texts.shape[0]
        total += bs
        texts = texts.to(DEVICE)
        labels = labels.to(DEVICE)
        
        output = model(texts)
        acc = accuracy(output, labels) * len(labels)
        pred = output.argmax(dim=1)
        all_predictions.append(pred)
        
        loss = criterion(output, labels) * len(labels)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if random.random() < 0.0015 and bs == 1:
            print("Input: "+' '.join([data_loader.dataset.idx2word[idx] for idx in texts[0].tolist() if idx not in {data_loader.dataset.word2idx[PAD], data_loader.dataset.word2idx[END]}]))
            print("Prediction:", pred.item(), '\tCorrect Output:', labels.item(), '\n')

    full_acc = 100*epoch_acc/total
    full_loss = epoch_loss/total
    print('[TEST]\t Loss: {:.4f}\t Accuracy: {:.2f}%'.format(full_loss, full_acc))
    predictions = torch.cat(all_predictions)
    return predictions, full_acc, full_loss


# In[18]:


if __name__=='__main__':
    evaluate(cnn_model, test_loader, criterion, use_tqdm=True) # Compute test data accuracy
    # pass


# # Step 4: Train a Recurrent Neural Network (RNN) [40 points]
# You will now build a text clasification model that is based on **recurrences**.

# ## <font color='red'>TODO:</font> Define the RNN Model [20 points]
# 
# First, you will define the RNN. As with the CNN, we provide you with the skeleton of the class, and you need to fill in parts of the `__init__(...)` and `forward(...)` methods. Each of these functions is worth 10 points.

# In[19]:


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, dropout, num_classes, pad_idx):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        ##### TODO #####

        # Create an embedding layer (https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
        #   to represent the words in your vocabulary. Make sure to use vocab_size, embed_size, and pad_idx here.
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.embed_size = embed_size
        # Create a recurrent network (use nn.GRU, not nn.LSTM) with batch_first = True
        # Make sure you use hidden_size, num_layers, dropout, and bidirectional here.
        self.GRU = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional).to(DEVICE)


        if bidirectional==True:
            self.D =2
        elif bidirectional==False:
            self.D=1

        # self.initial_state_h0 = 0


        # Create a dropout layer (nn.Dropout) using dropout
        self.dropout_layer = nn.Dropout(dropout).to(DEVICE)

        # Define a linear layer (nn.Linear) that consists of num_classes units
        #   and takes as input the output of the last timestep. In the bidirectional case, you should concatenate
        #   the output of the last timestep of the forward direction with the output of the last timestep of the backward direction).

        self.dense_layer = nn.Linear(hidden_size*self.D, num_classes).to(DEVICE)



    def forward(self, texts):
        """
        texts: LongTensor [batch_size, MAX_LEN]
        
        Returns output: Tensor [batch_size, num_classes]
        """
        ##### TODO #####
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Pass texts through your embedding layer to convert from word ids to word embeddings
        #   Resulting: shape: [batch_size, max_len, embed_size]
        texts = texts.type(torch.int64)
        # print('Content of embedding:', texts)
        # print('Shape of embedding:', texts.shape, '\n')
        # print('Type of embedding:', texts.dtype, '\n')

        final_embedding = (self.embedding(texts))
        final_embedding_gpu = final_embedding.to(device)
        # print(f"  final_embedding_gpu: {final_embedding_gpu.shape}, final_embedding_gpu dtype: {final_embedding_gpu.dtype}, final_embedding_gpu device: {final_embedding_gpu.get_device()}")

        # print('Content of embedding:', final_embedding)
        # print('Shape of embedding:', final_embedding.shape, '\n')
        # print('Type of embedding:', final_embedding.dtype, '\n')
        batch_size, max_len = texts.shape
        # print(f"batch_size: {batch_size}, max_len: {max_len}")
        # print(f"(batch_size, max_len, self.hidden_size): ({batch_size}, {max_len}, {self.hidden_size})")

        initial_state_h0 = torch.nn.parameter.Parameter(torch.randn(self.D*self.num_layers, batch_size, self.hidden_size)).to(device)
        # print(f"  initial_state_h0: {initial_state_h0.shape}, initial_state_h0 dtype: {initial_state_h0.dtype}, initial_state_h0 device: {initial_state_h0.get_device()}")

        # gru_input = torch.randn(batch_size, max_len, self.hidden_size).to(device)
        # print(f"  gru_input: {gru_input.shape}, gru_input dtype: {gru_input.dtype}, gru_input device: {gru_input.get_device()}")
        # # h_out = 32



        # Pass the result through your recurrent network
        #   See PyTorch documentation for resulting shape for nn.GRU
        output, hn = self.GRU(final_embedding_gpu, initial_state_h0)
        # print(f"  output: {output.shape}, output dtype: {output.dtype}")
        # print(f" hn: {hn.shape}, hn dtype: {hn.dtype}")


        # Concatenate the outputs of the last timestep for each direction (see torch.cat(...))
        #   This depends on whether or not your model is bidirectional.
        #   Resulting shape: [batch_size, num_dirs*hidden_size]
        # concatenated_output = torch.cat([h for h in output], dim=0)
        concatenated_output = output[:, -1, :]
        # print(f"concatenated_output: {concatenated_output.shape}, concatenated_output dtype: {concatenated_output.dtype}")

        # batch_size: 1, max_len: 150
        # (batch_size, max_len, self.hidden_size): (1, 150, 32)
        #   gru_input: torch.Size([1, 150, 16]), gru_input dtype: torch.float32
        #   initial_state_h0: torch.Size([4, 1, 32]), initial_state_h0 dtype: torch.float32
        #   output: torch.Size([1, 150, 64]), output dtype: torch.float32
        #  hn: torch.Size([4, 1, 32]), hn dtype: torch.float32
        #  concatenated_output: torch.Size([4, 32]), concatenated_output dtype: torch.float32
        #  dropout: torch.Size([4, 32]), dropout dtype: torch.float32

        # Apply dropout
        dropout = self.dropout_layer(concatenated_output)
        # print(f" dropout: {dropout.shape}, dropout dtype: {dropout.dtype}")

        # Pass your output through the linear layer and return its output 
        #   Resulting shape: [batch_size, num_classes]
        linear_output = self.dense_layer(dropout)
        # print(f" linear_output: {linear_output.shape}, linear_output dtype: {linear_output.dtype}")


        ##### NOTE: Do not apply a sigmoid or softmax to the final output - done in training method!


        return linear_output


# ##Sanity Check: RNN Model
# 
# The code below runs a sanity check for your `RNN` class. The tests are similar to the hidden ones in Gradescope. However, note that passing the sanity check does <b>not</b> guarantee that you will pass the autograder; it is intended to help you debug.

# In[20]:


### DO NOT EDIT ###

if __name__ == '__main__':
    # Test init
    inputs = [{'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 256, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 256, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 256, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 256, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 256, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 256, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 256, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 16, 'hidden_size': 256, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 256, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 256, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 256, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 256, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 256, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 256, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 256, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0}, {'vocab_size': 1000, 'embed_size': 64, 'hidden_size': 256, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0}]
    expected_outputs = [44546, 44676, 27202, 27268, 82178, 82308, 39874, 39940, 1620610, 1621636, 621698, 622212, 3986050, 3987076, 1411202, 1411716, 101762, 101892, 79810, 79876, 139394, 139524, 92482, 92548, 1742338, 1743364, 706562, 707076, 4107778, 4108804, 1496066, 1496580]

    sanityCheckModel(inputs, RNN, expected_outputs, "init", None)
    print()

    # Test forward
    inputs = [{'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 32, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 2, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 2, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 4, 'bidirectional': True, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 2, 'pad_idx': 0, 'batch_size': 2}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 1}, {'vocab_size': 29730, 'embed_size': 16, 'hidden_size': 64, 'num_layers': 4, 'bidirectional': False, 'dropout': 0, 'num_classes': 4, 'pad_idx': 0, 'batch_size': 2}]
    expected_outputs = [torch.Size([1, 2]), torch.Size([2, 2]), torch.Size([1, 4]), torch.Size([2, 4]), torch.Size([1, 2]), torch.Size([2, 2]), torch.Size([1, 4]), torch.Size([2, 4]), torch.Size([1, 2]), torch.Size([2, 2]), torch.Size([1, 4]), torch.Size([2, 4]), torch.Size([1, 2]), torch.Size([2, 2]), torch.Size([1, 4]), torch.Size([2, 4]), torch.Size([1, 2]), torch.Size([2, 2]), torch.Size([1, 4]), torch.Size([2, 4]), torch.Size([1, 2]), torch.Size([2, 2]), torch.Size([1, 4]), torch.Size([2, 4]), torch.Size([1, 2]), torch.Size([2, 2]), torch.Size([1, 4]), torch.Size([2, 4]), torch.Size([1, 2]), torch.Size([2, 2]), torch.Size([1, 4]), torch.Size([2, 4])]
    sanity_dataset = TextDataset(train_data, 'train', 5, 150)
    sanity_loader = torch.utils.data.DataLoader(sanity_dataset, batch_size=50, shuffle=True, num_workers=2, drop_last=True)

    sanityCheckModel(inputs, RNN, expected_outputs, "forward", sanity_loader)


# ## Train RNN Model
# First, we initialize the train and test dataloaders.

# In[21]:


if __name__=='__main__':
    THRESHOLD = 5 # Don't change this
    MAX_LEN = 200 # Don't change this
    BATCH_SIZE = 64 # Feel free to try other batch sizes

    train_dataset = TextDataset(train_data, 'train', THRESHOLD, MAX_LEN)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    test_dataset = TextDataset(test_data, 'test', THRESHOLD, MAX_LEN, train_dataset.idx2word, train_dataset.word2idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


# Now you can instantiate your model. We provide you with some recommended hyperparameters; you should be able to get the desired accuracy with these, but feel free to play around with them.

# In[22]:


if __name__=='__main__':
    rnn_model = RNN(vocab_size = train_dataset.vocab_size, # Don't change this
                embed_size = 128, 
                hidden_size = 128, 
                num_layers = 2,
                bidirectional = True,
                dropout = 0.5,
                num_classes = 2, # Don't change this
                pad_idx = train_dataset.word2idx[PAD]) # Don't change this

    # Put your model on device
    rnn_model = rnn_model.to(DEVICE)

    print('The model has {:,d} trainable parameters'.format(count_parameters(rnn_model)))


# Here, we create the criterion and optimizer; as with the CNN, we use cross-entropy loss and Adam optimization.

# In[23]:


if __name__=='__main__':    
    LEARNING_RATE = 6e-4 # Feel free to try other learning rates

    # Define your loss function
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Define your optimizer
    optimizer = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)


# Finally, we can train the model. We use the same `train_model(...)` function that we defined for the CNN. If the model is implemented correctly and you're using the GPU, this cell should take around <b>2 minutes</b> (or less). Feel free to change the number of epochs.

# In[24]:


if __name__=='__main__':    
    N_EPOCHS = 7 # Feel free to change this
    
    # train model for N_EPOCHS epochs
    train_model(rnn_model, N_EPOCHS, train_loader, optimizer, criterion)


# ## Evaluate RNN Model [20 points]
# 
# Now we can evaluate the RNN. 
# 
# To pass the autograder for the RNN, you will need to achieve **82% accuracy** on the hidden test set on Gradescope. Note that the Gradescope test set is very similar, and the accuracies between the two datasets should be comparable.

# In[25]:


if __name__=='__main__':    
    evaluate(rnn_model, test_loader, criterion, use_tqdm=True) # Compute test data accuracy


# # What to Submit
# 
# To submit the assignment, download this notebook as a <TT>.py</TT> file. You can do this by going to <TT>File > Download > Download .py</TT>. Then (optionally) rename it to `hwk2.py`.
# 
# You will also need to save the `cnn_model` and `rnn_model`. You can run the cell below to do this. After you save the files to your Google Drive, you need to manually download the files to your computer, and then submit them to the autograder.
# 
# You will submit the following files to the autograder:
# 1.   `hwk2.py`, the download of this notebook as a `.py` file (**not** a `.ipynb` file)
# 1.   `cnn.pt`, the saved version of your `cnn_model`
# 1.   `rnn.pt`, the saved version of your `rnn_model`

# In[26]:


### DO NOT EDIT ###

if __name__=='__main__':
    # from google.colab import drive
    # drive.mount('/content/drive')
    print()

    try:
        cnn_model is None
        cnn_exists = True
    except:
        cnn_exists = False

    try:
        rnn_model is None
        rnn_exists = True
    except:
        rnn_exists = False

    if cnn_exists:
        print("Saving CNN model....") 
        # torch.save(cnn_model, "drive/My Drive/cnn.pt")
        torch.save(cnn_model, "saved_models/cnn.pt")
    if rnn_exists:
        print("Saving RNN model....") 
        # torch.save(rnn_model, "drive/My Drive/rnn.pt")
        torch.save(rnn_model, "saved_models/rnn.pt")
    print("Done!")


# In[26]:





# In[26]:





# In[26]:




