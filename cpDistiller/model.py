import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from sklearn.decomposition import PCA

class GumbelSoftmax(nn.Module):

  def __init__(self, input_dim, category_num, eps =1e-20):
    super(GumbelSoftmax, self).__init__()
    self.logits = nn.Linear(input_dim, category_num)            
    self.input_dim = input_dim
    self.c_num = category_num
    self.eps = eps

  def gumbel_softonehot(self, logits, temperature):
    U = torch.rand(logits.shape)
    U = U.to(logits.device)
    y = logits -torch.log(-torch.log(U+self.eps)+self.eps)
    return F.softmax(y / temperature, dim=-1)

  def gumbel_onehot(self, logits, temperature):
    soft_one_hot = self.gumbel_softonehot(logits, temperature)
    shape = soft_one_hot.size()
    _, index = soft_one_hot.max(dim=-1)
    one_hot = torch.zeros_like(soft_one_hot).view(-1, shape[-1])
    one_hot.scatter_(1, index.view(-1, 1), 1)
    one_hot = one_hot.view(*shape)
    one_hot = (one_hot - soft_one_hot).detach() + soft_one_hot
    return one_hot 
  
  def forward(self, x, temperature=1.0, hard=False):
    
    logits = self.logits(x).view(-1, self.c_num)
    prob = F.softmax(logits, dim=-1)
    if not hard:
        y =  self.gumbel_softonehot(logits, temperature)
    else:  
        y = self.gumbel_onehot(logits, temperature)
    return logits, prob, y


class Gaussian(nn.Module):
  def __init__(self, input_dim, z_dim):
    super(Gaussian, self).__init__()
    
    self.mu = nn.Sequential(nn.Linear(input_dim, z_dim))
                                             
    self.var = nn.Sequential(nn.Linear(input_dim, z_dim),
                             nn.Softplus(),
                             )
  def forward(self, x):
    mu = self.mu(x)
    var = self.var(x)
    std = torch.sqrt(var+1e-10) 
    normal = torch.randn_like(std)
    z = mu + normal* std
    return mu, var, z 

class GM_Encoder(nn.Module):
    def __init__(self, input_dim,hidden_dim, z_dim,category_num):
        super(GM_Encoder, self).__init__()
         # q(category|x)
        self.qy_x = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),  
        )
        self.GumbelSoftmax = GumbelSoftmax(hidden_dim, category_num)

        # q(z|category,x)
        self.qz_yx = nn.Sequential(
            nn.Linear(input_dim + category_num, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
                
        )
        self.Gaussian =  Gaussian(hidden_dim, z_dim)
        self.projection_head = nn.Sequential(nn.Linear(z_dim,z_dim),
                                              nn.LeakyReLU()
                                               )
      
       
        
    def forward(self, x, temperature=1.0, hard=0):
        logits, prob, y = self.GumbelSoftmax(self.qy_x(x),temperature,hard)
        
        catxy =  torch.cat((x,y), dim=1)  
        grl = self.qz_yx(catxy)
        mu, var, z = self.Gaussian(grl)

        project_head = self.projection_head(z)
        output = {'mu': mu, 'var': var, 'gaussian': z, 
                'logits': logits, 'prob': prob, 'category': y, 'grl':grl,'project_head': project_head}
       
        return output
    
class GM_Decoder(nn.Module):
  def __init__(self, input_dim,hidden_dim, z_dim, category_num):
    super(GM_Decoder, self).__init__()

    # p(z|category_num)
    self.y_mu = nn.Sequential(nn.Linear(category_num, z_dim),)         
    self.y_var = nn.Sequential(nn.Linear(category_num, z_dim),nn.Softplus())
          

    # p(x|z)
    self.px_z = nn.Sequential(
        nn.Linear(z_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim,input_dim)
    )


  def forward(self, z, y):
    # p(z|category_num)
    y_mu = self.y_mu(y)
    y_var = self.y_var(y)
    
    # p(x|z)
    x_rec = self.px_z(z)

    output = {'y_mu': y_mu, 'y_var': y_var, 'x_rec': x_rec}
    return output



class Discriminator(nn.Module):
    def __init__(self, num_inputs,label_num):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
        nn.Linear(num_inputs, 128), 
        nn.LeakyReLU(),
        nn.Linear(128, 128), 
        nn.LeakyReLU(),
        nn.Linear(128, label_num)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class GMVAE(nn.Module):

    def __init__(self, input_dim,hidden_dim,out_dim,category_num):
        super(GMVAE, self).__init__()
        
        self.encode = GM_Encoder(input_dim,hidden_dim,out_dim,category_num)
        self.decode = GM_Decoder(input_dim,hidden_dim,out_dim,category_num)
        
    def forward(self, x,temperature=1.0, hard=0):
        
        encode_dict = self.encode(x, temperature, hard)
        z, y = encode_dict['gaussian'], encode_dict['category']
        decode_dict = self.decode(z, y)
        output = encode_dict
        for key, value in decode_dict.items():
            output[key] = value
        return output
      
def reshape_vector_to_matrix(vector, n):
    m = vector.shape[1]
    sqrt_m = np.sqrt(m)
    if sqrt_m.is_integer():
        reshaped_matrix = vector.reshape((n, 1, int(sqrt_m), int(sqrt_m)))
    else:
        raise ValueError(f"The second dimension of the vector ({m}) cannot be square rooted into an integer.")
    return reshaped_matrix


class EfficientAttention(nn.Module):         
    def __init__(self, c, b=1, gamma=2):
        super(EfficientAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return out * x
    
class PCAAttentionBlock(nn.Module):
    def __init__(self,input_dim=11664,output_dim=32,attetion=EfficientAttention(c=1)):
        super(PCAAttentionBlock, self).__init__()
        self.attention = attetion
        self.pca = PCA(n_components=output_dim) 
        self.attetion_linear = nn.Linear(input_dim, output_dim)
        self.attention_activation = nn.ReLU()
        self.output_linear = nn.Linear(output_dim,input_dim)
        self.output_activation = nn.ReLU()


    def forward(self, x):
        
        pca_x = self.pca.fit_transform(x.cpu().numpy())
        pca_x_torch = torch.from_numpy(pca_x).float().to(x.device)
        reshaped_matrix_example = reshape_vector_to_matrix(x, x.shape[0])
        attetion_x = self.attention(reshaped_matrix_example) 
        attetion_x = attetion_x.reshape(x.shape[0], -1) 
        attetion_x = self.attetion_linear(attetion_x)
        attetion_x = self.attention_activation(attetion_x).to(x.device)
        output = attetion_x + pca_x_torch 
        out_put = self.output_activation(self.output_linear(output))
        return out_put
        
class GMVAE_DL(nn.Module):

    def __init__(self, input_dim, hidden_dim,out_dim,category_num,pic_dim):
        super(GMVAE_DL, self).__init__()
        self.pic_dim = pic_dim
        self.pca_attention = PCAAttentionBlock(input_dim=pic_dim,attetion=EfficientAttention(c=1))
        self.encode = GM_Encoder(input_dim,hidden_dim,out_dim,category_num)
        self.decode = GM_Decoder(input_dim,hidden_dim,out_dim,category_num)
        
    def forward(self, x, temperature=1.0, hard=0):
        x_dl = x[:,-self.pic_dim:]
        if x.shape[0] >= 32:
            x_dl_attetionpca = self.pca_attention(x_dl)
        else:
            x_dl_attetionpca = x_dl
        x = x[:,:-self.pic_dim]
        x = torch.cat((x, x_dl_attetionpca), dim=1)  
        encode_dict = self.encode(x, temperature, hard)
        z, y = encode_dict['gaussian'], encode_dict['category']
        decode_dict = self.decode(z, y)
        output = encode_dict
        for key, value in decode_dict.items():
            output[key] = value
            
        return output
      