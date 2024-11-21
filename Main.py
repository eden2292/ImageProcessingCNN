import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Functions import load_images
from CustomLayers import *
from Kernels import *

# set paths to the images and CSV containing all labels. 
dataset_path = r'A:\Documents\Uni\preprocessed_img'
label_path = r'A:\Documents\Uni\ImageProcessingAssignment\labels.csv'

# define filter combinations for CustomConv2D
kernel_combo_1 = np.stack([high_pass_kernel, 
                           sharpen_kernel,
                           sobel_kernel_x, 
                           sobel_kernel_y, 
                           prewitt_kernel_x, 
                           prewitt_kernel_y, 
                           laplacian_kernel,
                           box_blur_kernel,
                           emboss_kernel,
                           ], axis=1)
kernel_combo_1 = np.expand_dims(kernel_combo_1, axis=-2)

# read the CSV, which contains image filepaths and labels. 
df = pd.read_csv(label_path)

# get file names and labels 
image_paths = df['file_path'].values 
labels = df['label'].values

# Initialize LabelEncoder to convert string labels to integers
label_encoder = LabelEncoder()

# Fit and transform the labels to integers
encoded_labels = label_encoder.fit_transform(labels)

# load in images from the dataset
x_data = np.array([load_images(img, dataset_path) for img in image_paths])

# Ensure the images are of type float32 and normalized between 0 and 1
x_data = x_data.astype('float32') / 255.0

# create an array of encoded labels
y_data = np.array(encoded_labels)

# split the data into test and train datasets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=None)

# Get the number of unique classes
num_classes = len(np.unique(y_data))

# Build a model using convolutional layers with Dropout to reduce overfitting
model = tf.keras.Sequential([
    # Input layer
    tf.keras.Input(shape=(64, 64, 1)),
    
    # Custom convolutional layers with batch normalization
    CustomConvLayer(gaussian_kernel),
    tf.keras.layers.BatchNormalization(),  # Added batch normalization
    CustomConv2D(filters=9, kernel_size=(3, 3), kernel_initializer=kernel_combo_1, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),  # Added batch normalization
    tf.keras.layers.ReLU(),
    
    # Custom edge detection layer
    CannyEdgeLayer(low=0.1, high=0.3),
    
    # Flatten and fully connected layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),  # Added batch normalization
    
    # Output layer
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
