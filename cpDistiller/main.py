import os
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Function
import random
from .losses import *
from .model import *
from scipy.optimize import fsolve
import logging
from typing import Union
from typing import Literal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.badatahmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
def equation(x,eps,n,k):
    return eps-eps*x**(k+1)-eps*x**(n-k)-1+x+eps*x

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.param = {}
        self.recent_param = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.param
                new_average = (1.0 - self.decay) * param.data + self.decay * self.param[name]
                self.param[name] = new_average.clone()

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.param
                self.recent_param[name] = param.data
                param.data = self.param[name]
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.recent_param
                param.data = self.recent_param[name]
        self.recent_param = {}

class cpDistiller_Model(object):
    def __init__(self, 
                 dataset, 
                 model_path,
                 epochs: int = 50, 
                 seed: int = 42, 
                 dim_hidden: int = 512, 
                 dim_out: int = 50,
                 lr: list = [1e-3,3e-3],
                 reduction: Literal['mean','sum'] = 'mean',
                 gpu_id: Union[None, int] = None,
                 category: int = 10,
                 gmvae_t: int = 1,
                 hard: int = 0,
                 pic_dim: int=11664,
                 eps: float = 1e-8,
                 name: str = 'result',
                 margin: int = 10,
                 decay_temp_rate: float = 0.0125,
                 min_temp: float = 0.5,
                 alpha: float = 0.75,
                 alp: list = [35,35],
                 print_step: int = 5,
                 use_mean: bool =True,
                 ema_alpha:float = 0.9,
                 seta: int = 1,
                 mo: Literal['cpDistiller-B','cpDistiller-C'] = 'cpDistiller-B'):
        """
        The main function of cpDistiller_Model is used for initializing the model.
        
        Parameters
        ----------
        dataset:
            Data handling interface for providing data for model training and testing.
        
        model_path:
            The path to save the model.
        
        epochs: int 
            The number of epochs for model training, default 50. 

        seed: int
            Seed value for random number generation, default 42.
                 
        dim_hidden: int 
            Hidden layer size of the model, default 256.

        dim_out: int 
            Dimension of the model's low-dimensional representation, default 50.

        lr: list 
            Learning rate list, representing the learning rate for the overall model and the learning rate for the discriminator separately, default [1e-3,3e-3].

        reduction: Literal['mean','sum']
            The computation method of the loss function, selectable as 'mean' or 'sum', default 'mean'.
        
        gpu_id: Union[None, int] 
            GPU ID used, default None.

        category: int 
            Prior number of Gaussians in the Gaussian mixture variational autoencoder, default 10.

        gmvae_t: int 
            Initial temperature coefficient for Gumbel-Softmax, default 1.

        hard: int 
            Calculation mode for Gumbel-Softmax, where 0 represents using soft labels and 1 represents using one-hot labels, default 0.

        pic_dim: int
            The dimensionality of image information extracted by the image model, default 144.

        eps: float
            The value of epsilon used when computing the ELBO (Evidence Lower Bound) loss, default 1e-8.

        name: str 
            The directory for saving the model, default 'result'.

        margin: int
            The margin value used in computing the triplet loss, default 10.

        decay_temp_rate: float 
            The decay rate of the temperature coefficient, default 0.0125.

        min_temp: float 
            The minimum temperature coefficient value.

        alpha: float 
            The weighting of soft labels at the positions of the ground truth labels.

        alp: list 
            The weighting for triplet loss and discriminator loss calculations, default [35,35]. This implies that the weights are all equal to input.dim[1]/35.

        print_step: int 
            The frequency of printing loss information, default 5. This means printing the loss information every 5 epochs.

        give_mean: bool 
            Give mean of distribution or sample from it. Defaults to False
        
        mo: Literal['cpDistiller-B','cpDistiller-C']
            If set to 'cpDistiller-C', only CellProfiler-based features are used. If set to 'cpDistiller-B' (default), both CellProfiler-based features and cpDistiller-extractor-based features are used for training. Alternatively, you can set the 'mo' to 'cpDistiller-C' but provide only cpDistiller-extractor-based features as input, effectively using a variant that does not rely on CellProfiler features (cpDistiller-D).
            
        """
        self.dataset = dataset
        self.epochs = epochs
        self.out_path = model_path
        self.lr = lr
        self.category = category
        # self.ema_alpha = 1-5/self.epochs
        self.ema_alpha = ema_alpha
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if gpu_id is not None:
            self.device = torch.device("cuda:{}".format(gpu_id))
        self.init_temp = gmvae_t
        self.gmvae_t = gmvae_t
        self.hard = hard
        self.name = name
        self.decay_temp_rate = decay_temp_rate
        self.min_temp = min_temp
        self.reduction = reduction
        self.alpha = alpha
        self.alp = alp
        self.print_step = print_step
        self.mod = self.dataset.mod
        self.seed = seed
        self.use_mean = use_mean
       
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.badatahmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

        self.D_row = Discriminator(dim_out,len(set(dataset.label_row))).to(device=self.device)
        self.D_col = Discriminator(dim_out,len(set(dataset.label_col))).to(device=self.device)
        
        self.triple_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=margin,reduction=self.reduction)
        self.LabelSmoothing = LabelSmoothingCrossEntropy(reduction=self.reduction).to(self.device)
        mse_loss = nn.MSELoss(reduction='none').to(device=self.device)
        self.elbo_loss = ELBOLoss(mse_loss,seta,eps,reduction=reduction).to(device=self.device)

       

        if mo == 'cpDistiller-C':  #  Alternatively, you can set the 'mo' to 'cpDistiller-C' but provide only cpDistiller-extractor-based features as input, effectively using a variant that does not rely on CellProfiler features (cpDistiller-D).
            self.input_dim = dataset.data.X.shape[1] 
            self.model = GMVAE(self.input_dim,dim_hidden,dim_out,category).to(
                    device=self.device)
        elif "cpDistiller-B":
            self.input_dim = dataset.data.X.shape[1] - pic_dim + int((int(math.sqrt(pic_dim))-9)/9+1)*int((int(math.sqrt(pic_dim))-9)/9+1)
            self.model = GMVAE_DL(self.input_dim,dim_hidden,dim_out,category,pic_dim).to(
                    device=self.device)

        self.loss_total = None
        row_sum = len(set(dataset.label_row))
        self.smooth_label_row = None
        for i in range(row_sum):
            q = fsolve(equation,0.3,args=(self.alpha,row_sum,i))
            vector = np.full(row_sum, self.alpha)
            distances= np.abs(np.arange(row_sum)-np.full(row_sum, i))
            vector = vector*q**distances
            vector /=vector.sum()
            if self.smooth_label_row is None:
                self.smooth_label_row = vector
            else:
                self.smooth_label_row = np.concatenate((self.smooth_label_row, vector), 0)
        self.smooth_label_row = self.smooth_label_row.reshape((row_sum,row_sum))

        col_sum = len(set(dataset.label_col))
        self.smooth_label_col = None
        for i in range(col_sum):
            q = fsolve(equation,0.3,args=(self.alpha,col_sum,i))
            vector = np.full(col_sum, self.alpha)
            distances= np.abs(np.arange(col_sum)-np.full(col_sum, i))
            vector = vector*q**distances
            vector /=vector.sum()
            if self.smooth_label_col is None:
                self.smooth_label_col = vector
            else:
                self.smooth_label_col = np.concatenate((self.smooth_label_col, vector), 0)
        self.smooth_label_col = self.smooth_label_col.reshape((col_sum,col_sum))

        if self.mod ==1:
            batch_sum = len(set(dataset.label_batch))
            self.smooth_label_batch = None
            for i in range(batch_sum):
                q = fsolve(equation,0.3,args=(self.alpha,batch_sum,i))
                vector = np.full(batch_sum, self.alpha)
                distances= np.abs(np.arange(batch_sum)-np.full(batch_sum, i))
                vector = vector*q**distances
                vector /=vector.sum()
                if self.smooth_label_batch is None:
                    self.smooth_label_batch = vector
                else:
                    self.smooth_label_batch = np.concatenate((self.smooth_label_batch, vector), 0)
            self.smooth_label_batch = self.smooth_label_batch.reshape((batch_sum,batch_sum))
            self.D_batch = Discriminator(dim_out,len(set(dataset.label_batch))).to(device=self.device)
            self.optimizer = torch.optim.AdamW([
            {'params': self.model.parameters(),'lr':self.lr[0]},
            {'params': self.D_row.parameters(),'lr':self.lr[1]},
            {'params': self.D_col.parameters(),'lr':self.lr[1]},
            {'params':self.D_batch.parameters(),'lr':self.lr[1]},
             ])
        else:
            self.optimizer = torch.optim.AdamW([
                {'params': self.model.parameters(),'lr':self.lr[0]},
                {'params': self.D_row.parameters(),'lr':self.lr[1]},
                {'params': self.D_col.parameters(),'lr':self.lr[1]},
            ])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        

    def load_model(self, 
                   model_file:str):
        """
        The function used to load model parameters.
        
        Parameters
        ----------

        model_file:str
            The path to load model parameters.

        """
        skpt_dict = torch.load(model_file, map_location=self.device)
        self.model.load_state_dict(skpt_dict)
          
    def train(self):
        """
        The function used to train model.
        
        """
        self.model.train()
        self.D_row.train()
        self.D_col.train()
        logging.info(self.model)
        logging.info(self.D_row)
        logging.info(self.D_col)
        if self.mod ==1:
            self.D_batch.train()
            logging.info(self.D_batch)
        self.ema = EMA(self.model, self.ema_alpha)
        self.ema.register()
        self.out_path = os.path.join(self.out_path,self.name)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        self.train_loss_list = []
        self.loss_ELBO_list = []
        self.loss_d_list = []
        self.loss_triple_list  = []
        len_dataloader = self.dataset.sum_num//self.dataset.batch_size 
        if self.mod==0:
            for epoch in range(self.epochs):
                category = {i:0 for i in range(self.category)}
                self.model.train()
                self.D_row.train()
                self.D_col.train()
                batch_num = 0
                train_data = self.dataset.train()
                epoch_loss = 0
                epoch_loss_elbo = 0
                epoch_loss_triple = 0
                epoch_loss_d = 0
                acc_row = 0
                acc_col = 0
                for anchor,positive,negative,row,col in train_data:
                    p = float(batch_num + epoch * len_dataloader) / self.epochs / len_dataloader
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    anchor = torch.FloatTensor(anchor).to(device=self.device)
                    positive = torch.FloatTensor(positive).to(device=self.device)
                    negative = torch.FloatTensor(negative).to(device=self.device)
                    row_smooth = torch.FloatTensor(self.smooth_label_row[row]).to(device=self.device)
                    col_smooth = torch.FloatTensor(self.smooth_label_col[col]).to(device=self.device)
                    row = torch.LongTensor(row).to(device=self.device)
                    col = torch.LongTensor(col).to(device=self.device)

                    out_info = self.model(anchor,self.gmvae_t,self.hard)
                    out_info_p = self.model(positive,self.gmvae_t,self.hard)
                    out_info_n = self.model(negative,self.gmvae_t,self.hard)

                    x = ReverseLayerF.apply(out_info['gaussian'], alpha)
                    drow = self.D_row(x)
                    dcol = self.D_col(x)
                
                    loss_triple = self.input_dim/ self.alp[1]*self.triple_loss(out_info['project_head'],out_info_p['project_head'],out_info_n['project_head'])
                   
                    loss_d =self.input_dim/self.alp[0]*(self.LabelSmoothing(drow, row,row_smooth) + self.LabelSmoothing(dcol, col,col_smooth))
                    loss_elbo = self.elbo_loss(out_info['x_rec'],out_info['X'],out_info['gaussian'], out_info['mu'], out_info['var'], out_info['y_mu'], out_info['y_var'],out_info['logits'], out_info['prob'])
                    self.loss_total =   loss_elbo  + loss_triple + loss_d

                    max_category = list(torch.max(out_info['category'],1)[1].cpu().data.numpy())
                    for i in range(self.category):
                        category[i]+=max_category.count(i)
                    epoch_loss +=self.loss_total.item()
                    epoch_loss_elbo += (loss_elbo.item())
                    epoch_loss_triple += loss_triple.item()
                    epoch_loss_d += loss_d.item()
                    self.optimizer.zero_grad()
                    self.loss_total.backward()
                    self.optimizer.step() 

                    batch_num += 1
                    acc_row += (drow.max(dim=1)[1] == row).float().sum().detach().cpu().numpy()
                    acc_col += (dcol.max(dim=1)[1] == col).float().sum().detach().cpu().numpy()

                self.scheduler.step()
                self.ema.update()
                self.gmvae_t = np.maximum(self.init_temp * np.exp(-self.decay_temp_rate * epoch), self.min_temp)   
                self.train_loss_list.append(epoch_loss / batch_num)
                self.loss_ELBO_list.append(epoch_loss_elbo/ batch_num)
                self.loss_triple_list.append(epoch_loss_triple / batch_num)
                self.loss_d_list.append(epoch_loss_d / batch_num)
                if epoch % self.print_step==0:
                    logging.info("epoch:{} average loss:{:.2f} ELBO:{:.2f} D loss:{:.2f} triplet loss:{:.2f} ".format(epoch , self.train_loss_list[-1],self.loss_ELBO_list[-1],self.loss_d_list[-1],self.loss_triple_list[-1]))
                    logging.info("acc row:{:.2f}% ,acc col:{:.2f}%".format(acc_row*100/sum(category.values()),acc_col*100/sum(category.values())))

        elif self.mod ==1:
            for epoch in range(self.epochs):
                category = {i:0 for i in range(self.category)}
                self.model.train()
                self.D_row.train()
                self.D_col.train()
                self.D_batch.train()
                batch_num = 0
                train_data = self.dataset.train()
                epoch_loss = 0
                epoch_loss_elbo = 0
                epoch_loss_triple = 0
                epoch_loss_d = 0
                acc_row = 0
                acc_col = 0
                acc_batch = 0
                for anchor,positive,negative,row,col,batch  in train_data:
                    p = float(batch_num + epoch * len_dataloader) / self.epochs / len_dataloader
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    anchor = torch.FloatTensor(anchor).to(device=self.device)
                    positive = torch.FloatTensor(positive).to(device=self.device)
                    negative = torch.FloatTensor(negative).to(device=self.device)
                    row_smooth = torch.FloatTensor(self.smooth_label_row[row]).to(device=self.device)
                    col_smooth = torch.FloatTensor(self.smooth_label_col[col]).to(device=self.device)
                    batch_smooth = torch.FloatTensor(self.smooth_label_batch[batch]).to(device=self.device)
                    row = torch.LongTensor(row).to(device=self.device)
                    col = torch.LongTensor(col).to(device=self.device)
                    batch = torch.LongTensor(batch).to(device=self.device)
                    
                    out_info = self.model(anchor,self.gmvae_t,self.hard)
                    out_info_p = self.model(positive,self.gmvae_t,self.hard)
                    out_info_n = self.model(negative,self.gmvae_t,self.hard)
                    
                    x = ReverseLayerF.apply(out_info['gaussian'], alpha)
                    drow = self.D_row(x)
                    dcol = self.D_col(x)
                    dbatch = self.D_batch(x)
                
                    loss_triple =  self.input_dim/self.alp[0]*self.triple_loss(out_info['project_head'],out_info_p['project_head'],out_info_n['project_head'])
                    loss_d = self.input_dim/self.alp[1]*(self.LabelSmoothing(dbatch,batch,batch_smooth) + self.LabelSmoothing(drow, row,row_smooth) + self.LabelSmoothing(dcol, col,col_smooth))
                
                    loss_elbo = self.elbo_loss(out_info['x_rec'],out_info['X'],out_info['gaussian'], out_info['mu'], out_info['var'], out_info['y_mu'], out_info['y_var'],out_info['logits'], out_info['prob'])
                    self.loss_total =   loss_elbo  + loss_triple + loss_d
                    max_category = list(torch.max(out_info['category'],1)[1].cpu().data.numpy())
                    for i in range(self.category):
                        category[i]+=max_category.count(i)
                    epoch_loss +=self.loss_total.item()
                    epoch_loss_elbo += (loss_elbo.item())
                    epoch_loss_triple += loss_triple.item()
                    epoch_loss_d += loss_d.item()
                    self.optimizer.zero_grad()
                    self.loss_total.backward()
                    self.optimizer.step() 
                    batch_num += 1
                    acc_row += (drow.max(dim=1)[1] == row).float().sum().detach().cpu().numpy()
                    acc_col += (dcol.max(dim=1)[1] == col).float().sum().detach().cpu().numpy()
                    acc_batch += (dbatch.max(dim=1)[1] == batch).float().sum().detach().cpu().numpy()

                self.scheduler.step()
                self.ema.update()
                self.gmvae_t = np.maximum(self.init_temp * np.exp(-self.decay_temp_rate * epoch), self.min_temp)   
                self.train_loss_list.append(epoch_loss / batch_num)
                self.loss_ELBO_list.append(epoch_loss_elbo/ batch_num)
                self.loss_triple_list.append(epoch_loss_triple / batch_num)
                self.loss_d_list.append(epoch_loss_d / batch_num)
                if epoch %self.print_step==0:
                    logging.info("epoch:{} average loss:{:.2f} ELBO:{:.2f} D loss:{:.2f} triplet loss:{:.2f} ".format(epoch , self.train_loss_list[-1],self.loss_ELBO_list[-1],self.loss_d_list[-1],self.loss_triple_list[-1]))
                    logging.info("acc batch:{:.2f}% ,acc row:{:.2f}% ,acc col:{:.2f}%".format(acc_batch*100/sum(category.values()),acc_row*100/sum(category.values()),acc_col*100/sum(category.values())))
           
        
           
        torch.save(self.model.state_dict(), os.path.join(self.out_path,'final_model.ckpt'))
        self.ema.apply()
        torch.save(self.model.state_dict(), os.path.join(self.out_path,'final_model_ema.ckpt'))
        self.ema.restore()

    def eval(self, data, batch_size: int = 256):
        """
        The function used to eval.

        Parameters
        ----------
        batchsize : int
            Batch size during evaling, default 256.

        """
        set_seed(self.seed)
        self.model.eval()
        self.D_row.eval()
        self.D_col.eval()
        if self.mod == 1:
            self.D_batch.eval()
        result_X = None
        category = None
        for x in self.dataset.eval(data, batch_size):
            X = Variable(torch.FloatTensor(x).to(device=self.device))
            with torch.no_grad():
                output = self.model(X,self.gmvae_t,self.hard)
                if self.use_mean:
                    output_x = output['mu'] 
                else:
                    output_x = output['gaussian'] 
                output_c = torch.max(output['category'],1)[1]
            if result_X is None:
                result_X = output_x.cpu().data.numpy()
                category = output_c.cpu().data.numpy()
            else:
                result_X = np.concatenate((result_X, output_x.cpu().data.numpy()), 0)
                category = np.concatenate((category, output_c.cpu().data.numpy()), 0)
                
        return result_X,category
