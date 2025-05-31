import anndata as ad
import numpy as np
import random
import os
import torch
import logging
from typing import Literal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataSet(object):
    
    
    def __init__(self,  data, 
                        seed: int=42,
                        batch_size: int=256,
                        mod: Literal[1,0]=0,
                        p: float=0.5
                        ): 
        """
        Data handling interface for providing data for model training and testing.
        
        Parameters
        ----------
        data: Anndata
            Jump data for training or testing.
            
        seed: int 
            Seed value for random number generation, default 42.

        batch_size: int
            Batch size during training, default 256.
        
        mod: Literal[1,0]
            The 'mod' variable must be either 0 or 1, where 0 represents correct row and column effects, and 1 represents correct triple effects.
        
        p: float=0.5
            Proportion of samples drawn from MNN-based neighbors versus KNN-based neighbors. A value of p = 0.5 indicates an equal balance between MNN and KNN sampling.
            
        """

        super(DataSet, self).__init__()
        self.p = p
        self.batch_size = batch_size
        self.mod = mod
        assert mod==0 or mod==1, f"The 'mod' variable must be either 0 or 1, where 0 represents correct row and column effects, and 1 represents correct triple effects."
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.badatahmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        if type(data)==str:
            try:
                self.data = ad.read_h5ad(data)
                logging.info("read:{}".format(data))
            except Exception as e:
                logging.info("read {} failed".format(data))
                logging.info("error:{}".format(e.args))
        else:
            self.data = data
       
        self.label_row = self.data.obs['row'].cat.codes.to_numpy()
        self.label_col = self.data.obs['col'].cat.codes.to_numpy()
        if self.mod == 1:
            self.label_batch = self.data.obs['batch'].cat.codes.to_numpy()
        self.sum_num = self.data.shape[0]
        self.control_choice = np.where(self.data.obs['control']=='negative')[0]
        self.other_choice = np.where(self.data.obs['control']!='negative')[0]
        
        self.dict_1 = {}
        self.dict_2 = {}
    
        self.cache_0 = []
        self.cache_1 = []
        self.cache_2 = []
        self.cache_3 = []
        self.cache_4 = []
      
    def train(self):
        self.input = np.random.choice(self.data.shape[0], self.data.shape[0],
                                                    replace=False)
        self.prepare_triplet()        
        return self.next_train_batch()

    def next_train_batch(self):
        start = 0
        end = self.batch_size
        if self.mod ==0 :
            while start < self.sum_num:
                anchor = self.data.X[self.input[start:end]]
                positive = self.data.X[self.positive[self.input[start:end]]]
                negative = self.data.X[self.negative[self.input[start:end]]]
                row = self.label_row[self.input[start:end]]
                col  = self.label_col[self.input[start:end]]
                yield anchor,positive,negative,row,col
                start = end
                end += self.batch_size
        elif self.mod==1:
            while start < self.sum_num:
                anchor = self.data.X[self.input[start:end]]
                positive = self.data.X[self.positive[self.input[start:end]]]
                negative = self.data.X[self.negative[self.input[start:end]]]
                row = self.label_row[self.input[start:end]]
                col  = self.label_col[self.input[start:end]]
                batch = self.label_batch[self.input[start:end]]
                yield anchor,positive,negative,row,col,batch
                start = end
                end += self.batch_size

    def prepare_triplet(self):
        self.positive = np.arange(self.sum_num, dtype=int)
        self.negative = np.zeros(self.sum_num, dtype=int)
        for i in range(self.sum_num):
            if self.data.obs['control'][i]=='negative': 
                self.positive[i]=np.random.choice(self.control_choice)
                if len(self.cache_1)!=self.sum_num:
                    negative_choice = np.where(self.data.obsp['matrix'][i,:] ==0)[0]
                    final_choice = np.intersect1d(negative_choice, self.other_choice)
                    if final_choice.shape[0]==0:
                        final_choice = negative_choice
                    self.dict_1[i] = len(self.cache_1)
                    self.cache_1.append(final_choice)
                    self.negative[i] = np.random.choice(final_choice)
                else:
                    self.negative[i] = np.random.choice(self.cache_1[self.dict_1[i]])
            else:
                if len(self.cache_2)!=self.sum_num:
                    positive_choice = np.where(self.data.obsp['matrix'][i,:] ==3 )[0]
                    if positive_choice.shape[0]==0:
                        positive_choice = np.where(self.data.obsp['matrix'][i,:] ==2 )[0]
                        if positive_choice.shape[0]==0:
                            positive_choice = np.where(self.data.obsp['matrix'][i,:] ==1 )[0]
                    self.dict_2[i] = len(self.cache_2)
                    self.cache_2.append(positive_choice)
                if len(self.cache_3)!=self.sum_num:
                    positive_choice = np.where(self.data.obsp['matrix'][i,:] ==2 )[0]
                    if positive_choice.shape[0]==0:
                        positive_choice = np.where(self.data.obsp['matrix'][i,:] ==1 )[0]
                    self.cache_3.append(positive_choice)
                if random.random()>self.p:
                    self.positive[i]=np.random.choice(self.cache_2[self.dict_2[i]])
                else: 
                    self.positive[i]=np.random.choice(self.cache_3[self.dict_2[i]])

                if len(self.cache_4)!=self.sum_num:
                    negative_choice = np.where(self.data.obsp['matrix'][i,:] ==0)[0]
                    final_choice = np.intersect1d(negative_choice, self.control_choice)
                    if final_choice.shape[0]==0:
                        final_choice = negative_choice
                    self.cache_4.append(final_choice)
                    self.negative[i] = np.random.choice(final_choice)
                else:
                    self.negative[i] = np.random.choice(self.cache_4[self.dict_2[i]])    
                    
       
    def eval(self,input_data,batch_size=256):
        start = 0
        end = batch_size
        while start < input_data.shape[0]:
            yield input_data.X[start:end]
            start = end
            end += batch_size
