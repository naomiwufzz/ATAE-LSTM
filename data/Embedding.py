
# coding: utf-8

# In[1]:


#coding:utf-8
import os
import numpy as np
import torch as t
from nltk.tokenize import word_tokenize
from torch import nn


# In[2]:


import sys
sys.path.append('D:\\Jupyter\\Python\\ATAE-LSTM')
import Ipynb_importer


# In[3]:


from config import opt


# In[4]:


class Emb(object):
    def __init__(self):
        # create and init the items below
        # self.embedding   string word ==> np.ndarray vector
        self.embedding = {}
        
        # load the pre-trained data
        self.root = opt.base_root + opt.embedding_root
        f = open(self.root, 'r', encoding='UTF-8')
        
        l = f.readline()
        have_opened = 1
        while(l != '' and have_opened<=opt.embedding_load):
            # l : "a 0.1 0.2 0.3 ..."
            if l[-1] == '\n':
                l = l[:-1]
            l = l.split(' ')
            if not len(l)==opt.hidden_size + 1:
                l = f.readline()
                continue
            
            # l[0]  : string word
            # l[1:] : list<string> vector
            self.embedding[l[0].lower()] = np.array(l[1:], dtype=float)
            
            if(len(self.embedding)==have_opened):
                print('Embedding : have input words : '+str(have_opened))
                have_opened *= 2
            l = f.readline()
            
        print('Embedding : have input words : '+str(have_opened))
        f.close()
        
        # create the items to modify and use dynamically below
        # self.dictionary    string word ==> int index
        # self.words         int index ==> string word
        # self.no_pretrained string word ==> int appearance
        self.dictionary = {}
        self.words = []
        self.no_pretrained = {}
        
        return
    
    def _get_dic_(self):
        return self.dictionary
    
    def _get_words_(self):
        return self.words
    
    def _make_layer_(self):
        weight = []
        for word in self.words:
            weight.append(self.embedding[word])
        weight.append(np.random.uniform(-opt.epsilon, opt.epsilon, opt.hidden_size))
        
        layer = nn.Embedding.from_pretrained(t.FloatTensor(weight), freeze=False)
        
        return layer
    
    def _add_word_(self, sentence):
        # para sentence : a string to be tokenized by nltk.tokenize.word_tokenize
        sentence = word_tokenize(sentence)
        for word in sentence:
            word = word.lower()
            if word in self.dictionary:
                continue
            if word in self.embedding:
                # add this word into self.dictionary and self.words
                self.dictionary[word] = len(self.words)
                self.words.append(word)
                assert len(self.dictionary) == len(self.words)
            else:
                # if this no-pretrained word arise for at least opt.word_independence times
                # set an indepent embedding for it
                if word not in self.no_pretrained:
                    self.no_pretrained[word] = 1
                else:
                    self.no_pretrained[word] += 1
                    if self.no_pretrained[word] >= opt.word_independence:
                        self.no_pretrained.pop(word)
                        self.dictionary[word] = len(self.words)
                        self.words.append(word)
                        assert len(self.dictionary) == len(self.words)
                        
                        # set an indepent embedding for it
                        # init from U(-ε,ε) 
                        self.embedding[word] = np.random.uniform(-opt.epsilon, opt.epsilon, opt.hidden_size)
        return


# In[5]:


emb = Emb()


# In[6]:


if __name__=='__main__':
    for i in range(20):
        emb._add_word_('All the appetizers and salads were fabulous, the steak was mouth watering and the pasta was delicious!!!')
        print(emb._get_dic_())

