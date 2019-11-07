
# coding: utf-8

# In[1]:


import ipdb
import sys
sys.path.append('D:\\Jupyter\\Python\\ATAE-LSTM')
import Ipynb_importer
from data.Embedding import emb
from data.AspClas_ import AspClas
from models.ATAE_LSTM import ATAE_LSTM
from utils.visualize import Visualizer
from config import opt
from tqdm import tqdm
from random import randint


# In[2]:


import torch as t
from torch.utils.data import DataLoader
from torchnet import meter
from torch.autograd import Variable


# In[3]:


def val(model, dataset):
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()
    confusion_matrix = meter.ConfusionMeter(3)
    
    # for ii, data in enumerate(dataloader):
    for data in dataset:
        text, aspect, sentiment = data
        val_input = (Variable(text, volatile=True), Variable(aspect, volatile=True))
        val_label = Variable(sentiment.type(t.LongTensor), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input).view(1, opt.classes)
        confusion_matrix.add(score.data, sentiment.type(t.LongTensor))
    
    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100.0 * (cm_value[0][0] + cm_value[1][1] + cm_value[2][2]) / (cm_value.sum())
    return confusion_matrix, accuracy


# In[4]:


vis = Visualizer(opt.env)


# In[5]:


# step1 data
test_data = AspClas(opt.test_data_root, test=True, train=False, debug=False)
train_data = AspClas(opt.train_data_root, test=False, train=True, debug=False)
test_dataloader = DataLoader(
    test_data,
    opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers
)
train_dataloader = DataLoader(
    train_data,
    opt.batch_size,
    shuffle=False,
    num_workers=opt.num_workers
)


# In[6]:


# step2 configure model
model = ATAE_LSTM()


# In[7]:


# step3 criterion and optimizer
criterion = t.nn.CrossEntropyLoss()
lr = opt.lr
optimizer = t.optim.Adam(
    model.parameters(),
    lr = lr,
    weight_decay = opt.weight_decay
)


# In[ ]:


# step4 meters
loss_meter = meter.AverageValueMeter()
confusion_matrix = meter.ConfusionMeter(3)
previous_loss = 1e100


# In[ ]:


# step5 train
for epoch in range(opt.max_epoch):
    loss_meter.reset()
    confusion_matrix.reset()
    
    #for ii, (text, aspect, sentiment) in tqdm(enumerate(train_dataloader), total=len(train_data)):
    # ii = 0
    
    # for text, aspect, sentiment in train_data:
        # ii += 1
    len_train_data = len(train_data)
    for ii in range(int(len_train_data/opt.batch_size)):
        target = []
        score = []
        for i in range(ii*opt.batch_size, (ii+1)*opt.batch_size):
            i = (i+randint(1,len_train_data))%len_train_data
            text, aspect, sentiment = train_data[i]
            # train model
            input = (Variable(text), Variable(aspect))
            target.append(Variable(sentiment))
            score.append(model(input).view(1, opt.classes))
        target = t.cat(target, dim=0)
        score = t.cat(score, dim=0)
        optimizer.zero_grad()
        loss = criterion(score, target)
        loss.backward()
        optimizer.step()
        
        # meters update and visualize
        loss_meter.add(loss.data[0])
        confusion_matrix.add(score.data, target.data)
        if ii%opt.print_freq == 0:
            vis.plot('loss', loss_meter.value()[0])
            vis.log("score:{score},target:{target}".format(
                score = score,
                target = target
            ))
            
    model.save()
    
    
    # validate and visualize
    val_cm, val_accuracy = val(model, test_data)
    
    vis.plot('val_accuracy', val_accuracy)
    vis.plot('lr', lr*1000)
    vis.log("epoch:{epoch},\nlr:{lr},\nloss:{loss},\ntrain_cm:{train_cm},\nval_cm:{val_cm}".format(
        epoch = epoch,
        loss = loss_meter.value()[0],
        val_cm = str(val_cm.value()),
        train_cm=str(confusion_matrix.value()),
        lr=lr
    ))
    
    # update learning rate
    if loss_meter.value()[0].item() >= previous_loss:
        lr = lr * opt.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    previous_loss = loss_meter.value()[0]

Wh
Wv
omega
Wp
Wx
embedding.weight
lstm.weight_ih_l0
lstm.weight_hh_l0
lstm.bias_ih_l0
lstm.bias_hh_l0
lin.weight
lin.bias