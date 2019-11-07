
# coding: utf-8

# In[1]:


#coding:utf-8
import torch as t
import time


# In[2]:


class BasicModule(t.nn.Module):
    '''
    封装了nn.Module
    提供save和load两个方法
    '''
    
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))
        return
    
    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(t.load(path))
        return
    
    def save(self, name=None):
        '''
        保存模型 默认使用“模型名字+时间”作为文件名
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name[8:-2] + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        t.save(self.state_dict(), name)
        return name


# In[3]:


class Flat(t.nn.Module):
    '''
    把输入reshape成 (batch_size, dim_length)
    '''
    
    def __init__(self):
        super(Flat, self).__init__()
        # self.size = size
        return
    
    def forward(self, x):
        return x.view(x.size(0), -1)

