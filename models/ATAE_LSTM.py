
# coding: utf-8

# In[1]:


#coding:utf-8
import torch as t
import numpy as np
from torch import nn
import ipdb


# In[2]:


import sys
sys.path.append('D:\\Jupyter\\Python\\ATAE-LSTM')
import Ipynb_importer
from config import opt
from data.Embedding import emb
from models.BasicModule import BasicModule
from data.AspClas_ import AspClas


# In[3]:


class ATAE_LSTM(BasicModule):
    def __init__(self):
        super(ATAE_LSTM, self).__init__()
        
        self.embedding = emb._make_layer_()
        
        self.lstm = nn.LSTM(opt.hidden_size*2, opt.hidden_size)
        for k in self.lstm.state_dict().keys():
            self.lstm.state_dict()[k].uniform_(-opt.epsilon, opt.epsilon)
        
        self.hidden=(
            # 三个参数分别为 num_layers, batch_size, hidden_size
            t.nn.Parameter(
                t.Tensor(
                    np.random.uniform(-opt.epsilon, opt.epsilon, opt.hidden_size)
                ).view(1,1,opt.hidden_size)
            ),
            t.nn.Parameter(
                t.Tensor(
                    np.random.uniform(-opt.epsilon, opt.epsilon, opt.hidden_size)
                ).view(1,1,opt.hidden_size)
            )
        )
        
        self.Wh = t.nn.Parameter(
            t.Tensor(
                np.random.uniform(-opt.epsilon, opt.epsilon, opt.hidden_size**2)
            ).view(opt.hidden_size,opt.hidden_size)
        )
        self.Wv = t.nn.Parameter(
            t.Tensor(
                np.random.uniform(-opt.epsilon, opt.epsilon, opt.hidden_size**2)
            ).view(opt.hidden_size,opt.hidden_size)
        )
        
        self.tanh = t.nn.Tanh()
        
        self.omega = t.nn.Parameter(
            t.Tensor(
                np.random.uniform(-opt.epsilon, opt.epsilon, opt.hidden_size*2)
            ).view(1,opt.hidden_size*2)
        )
        self.softmax1 = nn.Softmax(dim=1)
        
        self.Wp = t.nn.Parameter(
            t.Tensor(
                np.random.uniform(-opt.epsilon, opt.epsilon, opt.hidden_size**2)
            ).view(opt.hidden_size,opt.hidden_size)
        )
        self.Wx = t.nn.Parameter(
            t.Tensor(
                np.random.uniform(-opt.epsilon, opt.epsilon, opt.hidden_size**2)
            ).view(opt.hidden_size,opt.hidden_size)
        )
        self.relu = t.nn.LeakyReLU()
        
        self.lin = nn.Linear(opt.hidden_size, opt.classes)
        for k in self.lin.state_dict().keys():
            self.lin.state_dict()[k].uniform_(-opt.epsilon, opt.epsilon)
        self.softmax2 = nn.Softmax(dim=0)
        
        return
    
    def forward(self, x):
        # word representation
        w = x[0]
        N = len(w)
        # aspect term
        v = x[1]
        # assert len(v)==1  # use mean()
        
        # e.g.
        # w torch.Size([16])
        # v torch.Size([1])
        
        e1 = self.embedding(x[0])
        e2 = self.embedding(x[1]).mean(dim=0).view(-1).expand(e1.size())
        # e.g.
        # e1 torch.Size([16, 300])
        # e2 torch.Size([1, 300]) -> torch.Size([16, 300])
        
        wv = t.cat((e1.view(N,1,opt.hidden_size), e2.view(N,1,opt.hidden_size)), dim=-1)
        # e.g.
        # wv torch.Size([16, 1, 600])
        
        out, (h, c) = self.lstm(wv, self.hidden)
        # e.g.
        # out torch.Size([16, 1, 300])
        # h torch.Size([1, 1, 300])
        # c torch.Size([1, 1, 300])
        
        Wh_H = self.Wh.mm(out.view(opt.hidden_size, N))
        Wv_Va_eN = self.Wv.mm(
            self.embedding(
                x[1]).mean(
                dim=0).view(
                opt.hidden_size, 1).expand(
                opt.hidden_size, N)
        )
        vh = t.cat((Wh_H, Wv_Va_eN), dim=0)
        # e.g.
        # Wh_H     torch.Size([300, 16])
        # Wv_Va_eN torch.Size([300, 16])
        # vh       torch.Size([600, 16])
        
        M = self.tanh(vh)
        # e.g.
        # M torch.Size([600, 16])
        
        alpha = self.softmax1(self.omega.mm(M))
        # e.g.
        # alpha torch.Size([1, 16])
        
        r = out.view(opt.hidden_size, N).mm(alpha.t())
        # e.g.
        # r torch.Size([300, 1])
        
        _h_ = self.relu(self.Wp.mm(r) + self.Wx.mm(h.view(opt.hidden_size,1)))
        # e.g.
        # _h_ torch.Size([300, 1])
        
        y = self.softmax2(self.lin(_h_.view(opt.hidden_size)))
        # e.g.
        # y torch.Size([3])
        
        return y


# In[4]:


if __name__=='__main__':
    get_ipython().run_line_magic('pdb', 'on')
    testDataset = AspClas(opt.base_root+'/data/restaurants-trial.xml')
    model = ATAE_LSTM()
    for text, aspect, sentiment in testDataset:
        x = (text, aspect)
        y = model(x)
    print(len(list(model.parameters())))

