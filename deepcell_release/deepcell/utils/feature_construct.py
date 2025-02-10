import sys

layer = 'C5_reduced'
import numpy as np
import tensorflow as tf
import os
os.environ['Cuda_VISIBLE_DEVICES'] = '1'
def tensor_max_pooling(input_tensor):
    max_pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(16, 16), strides=(16, 16))
    output_tensor = max_pooling_layer(input_tensor)
    return output_tensor
def reconstruct_image(tiled_images, num_tiles, overlap_ratio):
    
    
    
    
    _, num_images, height, width, channels = tiled_images.shape
    
    overlap = int(overlap_ratio * height)
   
    new_height = num_tiles * height - (num_tiles - 1) * overlap
    new_width = num_tiles * width - (num_tiles - 1) * overlap
    reconstructed_images = np.zeros((num_images, new_height, new_width, channels))
    
    for img_idx in range(num_images):
        reconstructed_image = np.zeros((new_height, new_width, channels))
        count_matrix = np.zeros((new_height, new_width, channels))
        for row in range(num_tiles):
            for col in range(num_tiles):
               
                tile = tiled_images[row * num_tiles + col, img_idx]
               
                start_row = row * (height - overlap)
                start_col = col * (width - overlap)
               
                reconstructed_image[start_row:start_row+height, start_col:start_col+width] += tile
                count_matrix[start_row:start_row+height, start_col:start_col+width] += 1
       
        reconstructed_images[img_idx] = reconstructed_image / count_matrix

    return reconstructed_images
def middle_features_untile(features, num_tiles=6, overlap_ratio = 1-0.75):
    batch_size = features.shape[1]
    num_images = features.shape[0] // (num_tiles ** 2) * batch_size 
    embeddings_initial_shape = 256
    if layer == 'conv5_block3_out':
        embeddings_initial_shape = 128
    elif layer == 'conv2_block3_out':
        embeddings_initial_shape = 256 * 4
    elif layer == 'conv3_block4_out':
        embeddings_initial_shape = 256 * 2
    elif layer == 'P4_merged':
        embeddings_initial_shape = 256 // 4
    elif layer == 'C5_reduced':
        embeddings_initial_shape = 256 // 16
    
    reshaped_tensor = features.reshape((num_tiles*num_tiles, num_images, 32, 32, embeddings_initial_shape))
    reconstructed_images = reconstruct_image(reshaped_tensor, num_tiles, overlap_ratio)
    print(reconstructed_images.shape)
    pooling_tensor = tensor_max_pooling(reconstructed_images)
    embeddings = np.reshape(pooling_tensor,(reconstructed_images.shape[0],-1))
    return embeddings