import os
import tensorflow as tf

# set paths to the images and CSV containing all labels. 
dataset_path = 'K:/preprocessed_img'
label_path = 'K:/ImageProcessingAssignment/labels.csv'

# function to load images from dataset
def load_images(image_name, dataset_path):
    image_path = os.path.join(dataset_path, image_name)

    # load image with tensorflow
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels = 1) # 1 channel as the image is greyscale. 

    return image