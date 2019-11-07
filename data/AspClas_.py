
# coding: utf-8

# In[1]:


#coding:utf8
import os
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import xml.etree.ElementTree as ET
import torch as t
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader


# In[2]:


import sys
sys.path.append('D:\\Jupyter\\Python\\ATAE-LSTM')
import Ipynb_importer
from config import opt
from data.Embedding import emb


# 改进版本的AspClas<br>
# 与之前的不同在于embedding的方式<br>
# 这里对单词序列使用序号进行编码<br>
# <center>
#                 string word =dict=> int index =t.nn.Embedding=> t.Tensor<br>
# </center>

# In[3]:


class AspClas(data.Dataset):
    
    def __init__(self, root, train=True, test=False, debug=False):
        '''
        主要目标：
        载入文件 解析数据
        根据 训练、验证、测试 划分数据
        '''
        # temporary list to save string format data
        # free after init
        self.raw_cases = []
        # list to save tensor format data
        # used when __getitem__() is called
        self.cases = []
        
        # dictionary used to transform polarity
        self.polar = {'positive':[2], 'neutral':[1], 'negative':[0]}
        
        # load root = 'restaurants-trial.xml'
        xml = ET.parse(root)
        for s in xml.findall('sentence'):
            if s.find('aspectTerms'):
                text = s.find('text').text
                asps = s.find('aspectTerms').findall('aspectTerm')
                for asp in asps: 
                    if asp.attrib['polarity'] in self.polar:
                        self.raw_cases.append((text, asp.attrib['term'], asp.attrib['polarity']))
        
        # division
        if test:
            pass
        elif train:
            self.raw_cases = self.raw_cases[:int(0.7*len(self.raw_cases))]
        else:
            self.raw_cases = self.raw_cases[int(0.7*len(self.raw_cases)):]
        
        # shuffle
        np.random.seed(100)
        self.raw_cases = np.random.permutation(self.raw_cases)
        
        # transform
        self._addall2embed_()
        self.transform(debug)
        
        
        return
    
    def __getitem__(self, index):
        '''
        一次返回一个 sentence-term-polarity
        '''
        return self.cases[index]
    
    def __len__(self):
        return len(self.cases)
    
    def _addall2embed_(self):
        for (raw_text, raw_term, raw_polarity) in self.raw_cases:
            emb._add_word_(raw_text)
            emb._add_word_(raw_term)
        return
    
    def transform(self, debug):
        '''
        transform the strings into word index
        transform the polar into one-hot of classes
        '''
        
        # refresh self.cases
        self.cases = []
        # dictionary used to transform text and term
        d = emb._get_dic_()
        
        # transform
        for (raw_text, raw_term, raw_polarity) in self.raw_cases:
            # transform text and term using emb.dictionary
            
            # text
            raw_words = word_tokenize(raw_text)
            text_index = []
            for rw in raw_words:
                rw = rw.lower()
                if rw in d:
                    text_index.append(d[rw])
                else:
                    if debug : print('un-pretrained word found : '+rw)
                    text_index.append(len(d))
            text_tensor = t.Tensor(text_index).long()
            
            # term
            raw_words = word_tokenize(raw_term)
            term_index = []
            for rw in raw_words:
                rw = rw.lower()
                if rw in d:
                    term_index.append(d[rw])
                else:
                    if debug : print('un-pretrained word found : '+rw)
                    term_index.append(len(d))
            term_tensor = t.Tensor(term_index).long()
            
            # transform polarity using self.polar
            
            # polarity
            polarity_tensor = t.Tensor(self.polar[raw_polarity]).long()
            
            self.cases.append(
                (
                    text_tensor,
                    term_tensor,
                    polarity_tensor
                )
            )
            
        self.raw_cases = []
        
        return


# In[4]:


if __name__=='__main__':
    testDataset = AspClas('restaurants-trial.xml', debug=False)
    testLoader = DataLoader(
        testDataset,
        batch_size = 2,
        shuffle = True,
        num_workers = 1
    )

