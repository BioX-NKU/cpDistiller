import os
import numpy as np
import pandas as pd
from .utils import get_absolute_image_path, replace_illumn, replace_s3_path, tiff_multiple_illum, merge_three
import multiprocessing
import warnings
import numpy as np
import tensorflow as tf
from deepcell.applications import Mesmer
import pandas as pd
from deepcell_toolbox.processing import histogram_normalization
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
save_path = './npz_data_demo' # Path where npz files are saved
illumn_path = './illumn_data_demo/' # Path for metadata
replace_path = '/data/pub/cell' # Path where files are downloaded
# example: download_path: s3://cellpainting-gallery/cpg0016-jump/source_4/images/2021_04_26_Batch1/images/BR00117035__2021-05-02T16_02_51-Measurement1/Images/
# example: loacl path:/data/pub/cell/images/2021_04_26_Batch1/images/2021_04_26_Batch1/images/BR00117035__2021-05-02T16_02_51-Measurement1/Images/
def process_illum(illum,
                part,
                total_parts=10,
               ):
    """
    Process a part of the dataset specified by the illum file to handle large datasets in segments.

    Parameters
    ----------
    illum: str
        The name of the illum file used to derive paths and for data processing.

    part: int
        The current segment index to process, starting from 0 up to total_parts - 1.

    total_parts: int
        The total number of segments the dataset is divided into (default is 10).

    Functionality:
    - Calculate the length of each segment of the data based on total_parts.
    - Read the specific segment of the data from the CSV file based on the part index.
    - Process images in the specified range, merging images from three channels (DNA, RNA, ER) and applying illumination correction.
    - Append processed images and label arrays (zeros) to respective arrays.
    - If processing the first segment, save directly. For other parts, load existing data, append new data, and save.

    Output:
        Data is saved into an npz file at a location derived from the illum parameter. If not the first part, data is appended to the existing file.
    """
    illum_path = os.path.join(illumn_path, illum)
    output_npz_name = illum[:-7] + '.npz'
    output_path = os.path.join(save_path, output_npz_name)

    initial_path = illum_path
    data = pd.read_csv(initial_path)

    part_length = len(data) // total_parts
    start_index = part * part_length
    end_index = (part + 1) * part_length if part < total_parts - 1 else len(data)

    X_array = np.empty((0, 1080, 1080, 2))
    y_array = np.empty((0, 1080, 1080))

    for index in range(start_index, end_index):
        row = data.iloc[index]
        
        DNA, RNA, ER = [get_absolute_image_path(replace_s3_path(row[f"PathName_Orig{channel}"], replace_path), row[f"FileName_Orig{channel}"]) for channel in ["DNA", "RNA", "ER"]]
        
        DNA_illum, RNA_illum, ER_illum = [get_absolute_image_path(replace_illumn(row[f"PathName_Illum{channel}"], replace_path), row[f"FileName_Illum{channel}"]) for channel in ["DNA", "RNA", "ER"]]
        
        X = merge_three(tiff_multiple_illum(DNA, DNA_illum), tiff_multiple_illum(RNA, RNA_illum), tiff_multiple_illum(ER, ER_illum))
        y = np.zeros((1080, 1080))

        X_array = np.append(X_array, [X], axis=0)
        y_array = np.append(y_array, [y], axis=0)

    if part == 0:
        np.savez(output_path, X=X_array, y=y_array)
    else:
        with np.load(output_path) as data:
            X_existing = data['X']
            y_existing = data['y']
            X_combined = np.append(X_existing, X_array, axis=0)
            y_combined = np.append(y_existing, y_array, axis=0)
        
        np.savez(output_path, X=X_combined, y=y_combined)

def multi_multi_process(illum):
    """
    Process the given illum file in multiple parts concurrently to leverage parallel processing capabilities.

    Parameters
    ----------
    illum:
        The name of the illum file that needs to be processed.


    Functionality:
    - Divides the processing of the illum file into 10 parts.
    - Sequentially initiates the processing of each part using the `process_illum` function.
    - This is intended to be used with parallel processing frameworks like multiprocessing to handle each part in a separate process.
    - The number of ten isused to leverage the usage of memory.
    """
    for part in range(10):
        process_illum(illum,part)

def tiff2npz(save_pth: str,
              illumn_pth: str,
              replace_pth :str,
              multiprocessing_num: int
            ):
    """
    This function is designed to process illumination files and generate corresponding numpy (npz) files containing image data. It leverages multiprocessing to handle large datasets efficiently.

    Parameters
    ----------
    save_path :str
        Path where npz files are saved

    illumn_path :str
        Path for metadata

    replace_path: str
        Path where files are downloaded

    """
    global save_path 
    save_path = save_pth
    global illumn_path 
    illumn_path = illumn_pth
    global replace_path 
    replace_path = replace_pth

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    all_files = sorted(os.listdir(illumn_path))
    illums = [item for item in all_files ]
    pool = multiprocessing.Pool(multiprocessing_num)
    
    pool.map(multi_multi_process,illums)
    pool.close()
    pool.join()

def power_transform(matrix, power=0.5):
    """
    Smooth the values of a matrix by applying a power transformation.
    Parameters
    ----------
        matrix: numpy.array
            The matrix to be smoothed.
            
        power:  float 
            The power parameter to control the degree of smoothing.

    Returns:
        numpy.array: The smoothed matrix.
    """
    # To avoid raising negative numbers and zero to a power, first shift the matrix so all elements are positive
    min_val = matrix.min()
    shifted_matrix = matrix - min_val + 1
    # Apply the power transformation
    smoothed_matrix = np.power(shifted_matrix, power)
    # Shift the matrix back to its original range
    return smoothed_matrix - 1 + min_val
def smooth_matrix_logsmooth_matrix(matrix, base=2):
    """
    Smooth the matrix using logarithmic transformation, increasing the degree of smoothing.
    Parameters
    ----------
        matrix: numpy.array 
            The matrix to be smoothed.
            
        scale_factor:   float 
            The scaling factor to enhance the logarithmic smoothing effect.
    Returns:
        numpy.array: The smoothed matrix.
    """
    # First multiply each element in the matrix by the scaling factor
    normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    # Apply the logarithmic transformation
    smoothed_matrix = np.power(normalized_matrix, 1 / base)
    return smoothed_matrix
def smooth_matrix_log(matrix, scale_factor=1):
    """
    Smooth the matrix using logarithmic transformation, increasing the degree of smoothing.
    Parameters
    ----------
        matrix: numpy.array 
            The matrix to be smoothed.
            
        scale_factor:   float 
            The scaling factor to enhance the logarithmic smoothing effect.
    Returns:
        numpy.array: The smoothed matrix.
    """
    # First multiply each element in the matrix by the scaling factor
    scaled_matrix = matrix * scale_factor
    # Apply the logarithmic transformation
    smoothed_matrix = np.log1p(scaled_matrix)
    return smoothed_matrix
def truncate_small_values(matrix, percentile=10):
    """
    Truncate values in the matrix that are below a certain percentile.
    Parameters
    ----------
        matrix: numpy.array 
            The input matrix.
            
        percentile: int 
            The percentile for truncation.
    Returns:
        numpy.array: The truncated matrix.
    """
    # Calculate the value at the specified percentile
    threshold = np.percentile(matrix, percentile)
    # Truncate values below the percentile
    truncated_matrix = np.where(matrix < threshold, threshold, matrix)
    return truncated_matrix
def truncate_big_values(matrix, percentile=10):
    """
    Truncate values in the matrix that are above a certain percentile.
    Parameters
    ----------
        matrix: numpy.array
            The input matrix.
            
        percentile: int
            The percentile for truncation.
    Returns:
        numpy.array: The truncated matrix.
    """
    # Calculate the value at the specified percentile
    threshold = np.percentile(matrix, 100-percentile)
    # Truncate values above the percentile
    truncated_matrix = np.where(matrix > threshold, threshold, matrix)
    return truncated_matrix

def npz2embedding(dir_path: str='./demo_embedding/',
                  data_paths: list= ['demo36.npz']):
    """
    The function processes data from specified paths, applying histogram normalization, smoothing, and merges processed channels. 
    It uses an instance of the Mesmer application from DeepCell to predict and extract embeddings, which are then saved into a CSV file.

    Parameters
    ----------
    dir_path: str 
        The address for storing the final embeddings.
        
    data_pahs: list
        The list of data to be processed.The data needs to be in NPZ format.
   
    """
    def save_embeddings(X_data, file_name="embeddings.csv", append=True):
        _, features = app.predict(X_data, image_mpp=0.5, compartment='nuclear')
        embeddings = tf.reshape(features, [X_data.shape[0] // 9,  -1])
        embeddings_array = embeddings.numpy()
        df = pd.DataFrame(embeddings_array)
        # Choose append mode or write mode based on the append parameter
        if append:
            df.to_csv(file_name, mode='a', header=False, index=False)
        else:
            df.to_csv(file_name, index=False)
    default_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    
    layer =  'C5_reduced'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Now set memory growth only for the GPUs that are needed
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set at the time of GPU device initialization
            print(e)
    # Iterate through the list of data paths and process each one
    for data_path in data_paths:
        try:
            # Set and print the file path
            file_path = data_path
            logging.info('file_path:{}'.format(file_path))
            
            # Construct the output file name
            output_name = dir_path+"demo_embeddings_"+os.path.splitext(os.path.basename(file_path))[0]+"_"+layer+".csv"
                    
            # Load the data and assign it to X_train and y
            with np.load(file_path) as data:
                X_train = data['X']
                y = data['y']

            # Record the number of rows in X_train
            num_rows = X_train.shape[0]
            
            # Perform histogram normalization on X_train

            X_train = histogram_normalization(X_train)
            
            # Smooth the first channel and set it as the first channel
            channel_1 = smooth_matrix_log(X_train[..., 0].flatten(), scale_factor=2) # first channel
            
            # Directly set the second channel
            channel_2 = X_train[..., 1].flatten()  # second channel

            # Merge the processed two channels back into X_train
            X_train = np.concatenate([channel_1.reshape(X_train.shape[0],1080,1080,-1), channel_2.reshape(X_train.shape[0],1080,1080,-1)], axis=-1)
            
            # Calculate statistical data
            # stats = {
            #     "Channel 1": {
            #         "min": np.min(channel_1),
            #         "max": np.max(channel_1),
            #         "mean": np.mean(channel_1),
            #         "variance": np.var(channel_1),
            #         "std": np.std(channel_1)
            #     },
            #     "Channel 2": {
            #         "min": np.min(channel_2),
            #         "max": np.max(channel_2),
            #         "mean": np.mean(channel_2),
            #         "variance": np.var(channel_2),
            #         "std": np.std(channel_2)
            #     }
            # }

            # Assume X_train is defined and app is a pre-configured model
            # rgb_images = create_rgb_image(X_train, channel_colors=['green', 'blue'])
            app = Mesmer()
            X_subset = X_train

            save_embeddings(X_subset, file_name=output_name, append=False)
        
        except Exception as e:
            continue
    os.environ["CUDA_VISIBLE_DEVICES"] = default_cuda_visible_devices


