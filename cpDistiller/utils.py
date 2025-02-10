import scanpy as sc
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import anndata as ad
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fill_missing_combinations(csv_file_path: str):
    """
    Fills missing well and site combinations in a CSV file with data from the smallest site number in the same well.

    Parameters
    ----------
    csv_file_path: str 
        The path to the CSV file containing the data.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    # Generate all possible combinations of wells and sites
    wells = [f"{chr(i)}{str(j).zfill(2)}" for i in range(ord('A'), ord('P')+1) for j in range(1, 25)]
    sites = range(1, 10)
    all_combinations = pd.DataFrame(list(itertools.product(wells, sites)), columns=['Metadata_Well', 'Metadata_Site'])
    # Find missing combinations in the existing data
    existing_combinations = df[['Metadata_Well', 'Metadata_Site']].drop_duplicates()
    missing_combinations = pd.merge(all_combinations, existing_combinations, on=['Metadata_Well', 'Metadata_Site'], 
                                    how='left', indicator=True)
    missing_combinations = missing_combinations[missing_combinations['_merge'] == 'left_only'].drop(columns=['_merge'])
    # For each missing combination, find the whole row data for the smallest site in the same well
    filled_rows = []
    for _, row in missing_combinations.iterrows():
        well = row['Metadata_Well']
        min_row = df[df['Metadata_Well'] == well].nsmallest(1, 'Metadata_Site').copy()
        min_row['Metadata_Site'] = row['Metadata_Site']  # Set the correct site for the missing combination
        filled_rows.append(min_row)
    # If there are filled rows, add them to the original DataFrame
    if filled_rows:
        filled_df = pd.concat(filled_rows, ignore_index=True)
        df = pd.concat([df, filled_df], ignore_index=True)
    # Output the processed DataFrame
    sorted_df = df.sort_values(by=['Metadata_Well', 'Metadata_Site'])
    # Save the sorted DataFrame to a CSV if needed
    sorted_df.to_csv(csv_file_path, index=False)

# def get_data_path(batchname:str,
#                   image_or_illum:str,
#                   BRR:str,
#                   file_name:str):
#     """
#     Constructs a full path to a data file based on provided parameters.

#     Parameters
#     ----------
#     batchname:  str
#         The batch name which is part of the directory structure.

#     image_or_illum: str
#         Subdirectory indicating type of data ('image' or 'illum').

#     BRR:   str 
#         Subdirectory name, generally a unique identifier.

#     file_name:  str
#         The name of the file including its extension.

#     Returns:    str 
#         The full path concatenated from the given components.
#     """
#     data_path = '/data/pub/cell/images'
#     path = data_path + os.path.sep + batchname + os.path.sep + image_or_illum + os.path.sep + BRR + os.path.sep + file_name
#     return path


def replace_s3_path_cpg0000(original_path: str, 
                            demo: str):
    """
    Replaces the download path to local path. It's written for cpg0000.

    Parameters
    ----------
    original_path:  str 
        The original file path.

    demo:   str
        local path.

    Returns: str
        The modified path with the new base or the original path.
    """
    parts = original_path.split('/')
    if len(parts) > 1:
        try:
            source_index = parts.index('images')  - 1
            parts[:source_index] = demo.split('/')
            return '/'.join(parts)
        except ValueError:
            return original_path
    else:
        return original_path
def replace_cpg0000_path(original_path:str,
                         demo:str):
    """
    Replaces the download path to local path.   It's written for cpg0000.

    Parameters
    ----------
    original_path: str 
        The original file path.

    demo: str
        local path.

    Returns:    str
        The modified path with the new base or the original path.
    """
    parts = original_path.split('/')
    if len(parts) > 1:
        try:
            source_index = parts.index('images')
            parts[:source_index] = demo.split('/')
            return '/'.join(parts)
        except ValueError:
            return original_path
    else:
        return original_path
def get_absolute_image_path(dir,
                            file_name):
    if dir[-1] != '/':
        dir += os.sep
    return dir + file_name


def replace_s3_path(original_path: str, 
                    demo: str):
    """
    Replaces the download path to local path. It's written for cpg0016.

    Parameters
    ----------
    original_path:  str
        The original file path.

    demo:   str
        local path.

    Returns:    str 
        The modified path with the new base or the original path.
    """
    parts = original_path.split('/')
    if len(parts) > 1:
        try:
            source_index = parts.index('Images') 
            parts[:source_index+1] = demo.split('/')
            return '/'.join(parts)
        except ValueError:
            return original_path
    else:
        return original_path
def replace_illumn(original_path,
                   demo):
    
    parts = original_path.split('/')
    demo_parts = demo.split('/')
    images_indices = [i for i, part in enumerate(demo_parts) if part == 'images']
    if len(parts) > 1:
        try:
            source_index = parts.index('illum')
            parts[:source_index] = demo_parts[:images_indices[1]]
            return '/'.join(parts)
        except ValueError:
            return original_path
    else:
        return original_path
    
def replace_illumn_cpg0000(original_path,
                           demo):
    parts = original_path.split('/')
    if len(parts) > 1:
        try:
            source_index = parts.index('illum') - 1
            parts[:source_index] = demo.split('/')
            return '/'.join(parts)
        except ValueError:
            return original_path
    else:
        return original_path
def tiff_multiple_illum(tiff_path: str,
                        illumn_path: str,
                        if_show: bool=False,
                        if_no_multi: bool=False):
    """
    Opens a TIFF image, applies an illumination correction, and optionally displays it.

    Args:

    tiff_path: str
        Path to the TIFF image file.

    illumn_path: str 
        Path to the numpy file containing illumination correction data.

    if_show: bool
        If True, display the image after applying illumination correction.

    if_no_multi: bool
        If True, return the original image array without applying illumination correction.

    Returns: ndarray
        The image array after applying illumination correction or the original array if `if_no_multi` is True.
    """
    with Image.open(tiff_path) as img:
        array = np.array(img)
    ill = np.load(illumn_path)
    if(if_show):
        plt.matshow(array * ill, cmap='gray')  
        plt.show()
    if(if_no_multi):
        return array
    return array * ill
def merge_three(dna_array,
                rna_array,
                er_array):
    """
    Merges three separate image arrays (representative of different stains or channels) into a single three-channel image.

    Args:

    dna_array (ndarray): 
        Array representing the DNA channel.
    rna_array (ndarray): 
        Array representing the RNA channel.
    er_array (ndarray): 
        Array representing the ER channel.

    Returns:
        ndarray: A new array with three channels, where the last channel is the average of the RNA and ER arrays.
    """
    return np.dstack((dna_array, (rna_array + er_array) / 2))

def scale_batch(data,
                key: str = 'batch'):
    """
    batch-wise normalization
    
    Parameters
    ----------
    data: Anndata
        Jump data is used for batch-wise normalization.
        
    key: str
        Seed value for random number generation, default 'batch'.

    Returns
    ----------
    adata: Anndata
        The data after batch-wise normalization.
            
    """
    batches = list(set(data.obs[key]))
    obs = data.obs_names
    scaled_adata = None
    for batch in batches:
        batch_adata = data[data.obs[key] == batch, :]
        data_matrix = batch_adata.X
        scaler = StandardScaler()
        scaled_data_matrix = scaler.fit_transform(data_matrix)
        batch_adata_scaled = sc.AnnData(X=scaled_data_matrix, obs=batch_adata.obs, var=batch_adata.var)
        if scaled_adata is None:
            scaled_adata = batch_adata_scaled
        else:
            scaled_adata = ad.concat([scaled_adata, batch_adata_scaled], join='outer', index_unique=None)
    adata = scaled_adata
    adata = adata[obs]
    return adata
def add_zero(attribute):
    if len(str(attribute)) == 1:
        return '0' + str(attribute)
    else:
        return str(attribute)
    
def extract_batch_number(filename):
    parts = filename.split('_')
    for part in parts:
        if 'Batch' in part:
            return int(part.replace('Batch', ''))
    return None

def merge_csv(path: str):
    """
    Concatenate data from multiple batches into a CSV file.
    
    Parameters
    ----------
    path: str
        Data path. In the directory, there are multiple CSV files, including several files named like 'Batch1.csv'.
        
    Returns
    ----------
    merged_df: 
        A CSV file containing multiple batches of data.
            
    """
    merged_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            batch_num = extract_batch_number(filename) 
            df = pd.read_csv(file_path)
            df['batch'] = str(batch_num)
            merged_df = pd.concat([merged_df, df], ignore_index=True)
    return merged_df

def merge_csv2h5ad(data_source,
                   adata):
    """
    Merge the CSV file with the AnnData file.
    
    Parameters
    ----------
    data_source: str
        Data path or data. This refers to a CSV file or the path of a CSV file.
        
    adata: Anndata
        The AnnData file to be merged.
    Returns
    ----------
    data: 
        A merged AnnData file.
            
    """
    if type(data_source)==str:
        try:
            data = pd.read_csv(data_source)
        except Exception as e:
            logging.info("error:{}".format(e.args))
    else:
        data = data_source
    data['Column_Label'] = data['Column_Label'].apply(add_zero)
    key = (adata.obs['Metadata_Plate'].astype(str)+adata.obs['Metadata_Well'].astype(str)).values
    data['key']=(data['BR_Code']+data['Row_Label']+data['Column_Label']).values
    data['key'] = pd.Categorical(data['key'], categories=key, ordered=True)
    all_categories_df = pd.DataFrame({'key': key})
    merged_df = pd.merge(all_categories_df, data, on='key', how='left').fillna(0)
    merged_df = merged_df.drop_duplicates(subset='key')
    columns_to_keep = [col for col in merged_df.columns if col not in  ['key','batch', 'Column_Label', 'BR_Code','Row_Label']]
    merged_df = merged_df[columns_to_keep]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(merged_df.values)
    data = ad.AnnData(
    X= np.concatenate((adata.X,X_scaled), axis=1),           
    obs=adata.obs.copy(), 
    )
    data.var_names = np.concatenate((adata.var_names, columns_to_keep))
    return data

def find_lose(source_folder='/data/pub/cell/'):
    """
    This function checks if any CSV files in the specified source folder do not have the expected number of rows.

    Parameters
    ----------
    source_folder:  str 
        The directory path where the CSV files are located.
    """
    # Target number of rows
    target_row_count = 3456
    # Loop through all files in the directory
    for file in os.listdir(source_folder):
        if file.endswith('.csv'):
            # Build the complete file path
            file_path = os.path.join(source_folder, file)
            # Read the CSV file
            try:
                df = pd.read_csv(file_path)
                # Check the number of rows
                if len(df) != target_row_count:
                    print(f"The file '{file}' does not have {target_row_count} rows. It has {len(df)} rows.")
            except Exception as e:
                print(f"Error reading {file}: {e}")
    print("Row count check complete.")

