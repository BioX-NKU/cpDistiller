import scanpy as sc
import numpy as np
import numpy as np
import hnswlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def get_nn(dataset1, 
           dataset2,
            names1, 
            names2, 
            knn: int = 10,
            type: str = 'mnn'):
    dim = dataset2.shape[1]
    num_elements = dataset2.shape[0]
    tree = hnswlib.Index(space="cosine", dim=dim)
    tree.init_index(max_elements=num_elements, ef_construction=200, M=32) 
    tree.set_ef(50)
    tree.add_items(dataset2)
    ind, _ = tree.knn_query(dataset1, k=knn)
    match = set()
    for a, b in zip(range(dataset1.shape[0]), ind):
        if type=='knn':
            for b_i in b[1:]: 
                match.add((names1[a], names2[b_i]))
        elif type=='mnn':
            for b_i in b[0:]: 
                match.add((names1[a], names2[b_i]))
    return match
       
def mnn(ds1,
         ds2,
         names1,
         names2, 
         knn: int = 5):     
    match1 = get_nn(ds1, ds2, names1, names2, knn=knn,type='mnn') 
    match2 = get_nn(ds2, ds1, names2, names1, knn=knn,type='mnn')  
    mutual = match1 & set([(b, a) for (a, b) in match2])  
    mutual = mutual | set([(b, a) for (a, b) in mutual])  
    return mutual
   
          
def batch_mnn(data,
              batch: str,
              Mnn: int = 10,
              nn_list: list = []):
    batch_list = list(set(data.obs[batch]))
    batch_list.sort()
    for i in range(len(batch_list)):
        for j in range(i+1,len(batch_list)):
            data1 = data[data.obs[batch]==batch_list[i]]
            data1_x = data1.obsm['X_pca']
            data1_name = data1.obs['name_num']
            data2 = data[data.obs[batch]==batch_list[j]]
            data2_x = data2.obsm['X_pca']
            data2_name = data2.obs['name_num']
            set_mnn = mnn(data1_x,data2_x,data1_name,data2_name,knn=Mnn)
            nn_list.append(set_mnn) 
def batch_knn(data,
              batch: str,
              Knn: int = 10,
              nn_list: list = []):
    batch_list = list(set(data.obs[batch]))
    batch_list.sort()
    for i in range(len(batch_list)):
        data1 = data[data.obs[batch]==batch_list[i]]
        data1_x = data1.obsm['X_pca']
        data1_name = data1.obs['name_num']
        set_knn = get_nn(data1_x ,data1_x ,data1_name ,data1_name ,knn=Knn,type='knn')
        nn_list.append(set_knn)    
          
def labeled(data,
            Mnn: int = 5,
            Knn: int = 10,
            technic_name_list: list = ['row','col']):
    """
    Calculate the knn and the mnn inter and intra technical effects to construct triplets.
    
    Parameters
    ----------
    data: Anndata
        Jump data for calculating knn and mnn.
        
    Mnn: int = 5
        The number of nearest neighbors considered when calculating MNN, default 5.

    Knn: int = 10
        The number of nearest neighbors considered when calculating KNN, default 10.

    technic_name_list: list
        The list of technic effect labels to be considered when calculating triplets, default ['row','col'].
            
    """
    matrix_apn =  np.eye(data.shape[0],dtype=bool)
    data.obs['name_num'] = np.arange(data.shape[0])
    sc.tl.pca(data)
    nn_list = []
    for batch in technic_name_list:
        batch_mnn(data,batch,Mnn,nn_list)
    nn_set = set()
    for set_ in nn_list:
        nn_set = nn_set | set_
    mnn_num = len(nn_set)
    logging.info('MNN pairs num: {}'.format(mnn_num//2))        
    nn_list = []
    for batch in technic_name_list:
        batch_knn(data,batch,Knn,nn_list)
    for set_ in nn_list:
        nn_set = nn_set | set_
    knn_num = len(nn_set)-mnn_num
    logging.info('KNN pairs num: {}'.format(knn_num))   
    tuple_ = np.array(list(nn_set))
    matrix_apn[tuple_[:,0],tuple_[:,1]]=True
    matrix_apn[tuple_[:,1],tuple_[:,0]]=True
    data.obsp['matrix']=matrix_apn.astype(float)
    

    
   