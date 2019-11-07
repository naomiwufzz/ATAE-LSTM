#coding:utf-8
import warnings
class DefaultConfig(object):
    env = 'default' # visdom环境
    model = 'ATAE-LSTM' # 使用的模型，名字与 models/__init__.py 中一致
    
    base_root = 'D:/Jupyter/Python/ATAE-LSTM/'
    embedding_root = 'data/glove.840B.300d/glove.840B.300d.trial' # 预训练词向量路径
    embedding_load = 20000 # 加载预训练词向量的量
    train_data_root = 'data/restaurants-trial.xml' # 训练集路径
    test_data_root = 'data/restaurants-trial.xml' # 测试集路径
    load_model_path = None # 加载预训练模型路径 None表示不加载
    
    word_independence = 5 # how many times an un-pretrained word have to appear for to be independent
    classes = 3 # how many classes
    
    batch_size = 4
    use_gpu = False
    hidden_size = 300 # the dimension of word vectors, aspect embeddings and the size of hidden layers
    num_workers = 1 # how many workers for loading data
    print_freq = -1 # how many batch as an interval between two prints
    
    max_epoch = 500
    lr = 0.01
    lr_decay = 0.9 # momentum
    weight_decay = 0.001 # L2-regularization
    epsilon = 0.01 # unknown parameters are randomly initialized from U(−ϵ,ϵ)
    
    def parse(self, kwargs):
        '''
        根据字典 kwargs 更新 config 参数
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning : opt has not attribute named %s" %k)
            setattr(self, k, v)
        
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
        return

opt = DefaultConfig()