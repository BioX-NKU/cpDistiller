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
                        mod: Literal[1,0]=0): 
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

            
        """

        super(DataSet, self).__init__()
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
            positive_choice = np.where(self.data.obsp['matrix'][i,:] >0 )[0]
            self.positive[i]=np.random.choice(positive_choice)
            negative_choice = np.where(self.data.obsp['matrix'][i,:] ==0)[0]
            self.negative[i] = np.random.choice(negative_choice)
       
    def eval(self,input_data,batch_size=256):
        start = 0
        end = batch_size
        while start < input_data.shape[0]:
            yield input_data.X[start:end]
            start = end
            end += batch_size
