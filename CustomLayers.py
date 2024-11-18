import tensorflow as tf
import numpy as np
from Kernels import *

# Define a custom layer for use with defined kernels
class CustomConvLayer(tf.keras.layers.Layer):
    def __init__(self, kernel):
        super(CustomConvLayer, self).__init__()
        # set the size of the kernel (3x3)
        # 3x3 selected as it takes less processing time, reduces overfitting, has higher spatial resolution
        self.kernel = tf.constant(kernel,dtype=tf.float32)
        # kernel shape - height, width, input channels (1 as greyscale), output channels (1 as greyscale)
        self.kernel = tf.reshape(self.kernel, [3, 3, 1, 1])

    # Apply custom kernel to the convolution layer. 
    def call(self, inputs):
        return tf.nn.conv2d(inputs, self.kernel, strides=1, padding='SAME')
    
# Define a custom 2D convolutional layer
class CustomConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, kernel_initializer = None, padding = 'valid', activation = None, **kwargs):
        super(CustomConv2D, self).__init__(**kwargs)
        # The number of filters for the layer
        self.filters = filters
        # the size of the kernel to be used
        self.kernel_size = kernel_size
        # determines if the image size is the same or reduced (will stay the same)
        self.padding = padding
        # any activations to occur after the convolution (I.E ReLU)
        self.activation = tf.keras.activations.get(activation)
        # used to initialize pre-defined kernels
        self.kernel_initializer = kernel_initializer 

    def build(self, input_shape):
        # define the shape of the kernel, including the number of channels (input_shape) and number of filters
        kernel_shape = (*self.kernel_size, input_shape[-1], self.filters)
        # has a custom kernel initializer been provided?
        if self.kernel_initializer is not None:
            # if yes, use it to create the kernel weights. 
            self.kernel = self.add_weight(name = "kernel",
                                          shape = kernel_shape,
                                          initializer = tf.constant_initializer(self.kernel_initializer),
                                          trainable = True)
        else:
            # if no, use the defauly glorot_uniform to create kernel weights. 
            self.kernel = self.add_weight(name = "kernel",
                                          shape = kernel_shape,
                                          initializer = "glorot_uniform",
                                          trainable = True)
        # define the bias weight for the layer.
        self.bias = self.add_weight(name = "bias",
                                    shape = (self.filters,),
                                    initializer = "zeros",
                                    trainable = True)
    
    def call(self, inputs):
        # perform 2D convolution. 
        output = tf.nn.conv2d(inputs, self.kernel, strides = 1, padding = self.padding.upper())
        # add the bias to the output of the convolution
        output = tf.nn.bias_add(output, self.bias)
        # apply any activation functions that have been set
        if self.activation is not None:
            output = self.activation(output)
        return output
    
# Define a custom layer for canny edge detection. 
class CannyEdgeLayer(tf.keras.layers.Layer):

    # low = lower threshold for hysteresis
    # high = upper threshold for hysteresis
    def __init__(self, low=0.1, high=0.3, **kwargs):
        super(CannyEdgeLayer, self).__init__(**kwargs)
        self.low = low
        self.high = high

    def build(self, input_shape):
        # define kernals for gaussian blurring and sobels. 
        self.sobel_x = sobel_kernel_x
        self.sobel_y = sobel_kernel_y
        # [height, width, number of channels (1 as greyscale image), output channels (1 as greyscale)]
        self.sobel_x = tf.constant(self.sobel_x, dtype= tf.float32)
        self.sobel_x = tf.reshape(self.sobel_x, [3, 3, 1, 1])
        self.sobel_y = tf.constant(self.sobel_y, dtype= tf.float32)
        self.sobel_y = tf.reshape(self.sobel_y, [3, 3, 1, 1])

        self.gaussian_kernel = tf.constant(gaussian_kernel, dtype = tf.float32)
        self.gaussian_kernel = tf.reshape(self.gaussian_kernel, [3, 3, 1, 1])
        super(CannyEdgeLayer, self).build(input_shape)

    def call(self, inputs):
        num_channels = tf.shape(inputs)[-1]

        # Adjust kernels for multi-channel input
        gaussian_kernel = tf.tile(self.gaussian_kernel, [1, 1, num_channels, 1])
        sobel_x = tf.tile(self.sobel_x, [1, 1, num_channels, 1])
        sobel_y = tf.tile(self.sobel_y, [1, 1, num_channels, 1])

        # Apply Gaussian blur
        blurred = tf.nn.depthwise_conv2d(inputs, gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME')

        # Compute gradients
        gradient_x = tf.nn.depthwise_conv2d(blurred, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        gradient_y = tf.nn.depthwise_conv2d(blurred, sobel_y, strides=[1, 1, 1, 1], padding='SAME')

        # calculate edge strength and orientation
        gradient_magnitude = tf.sqrt(tf.square(gradient_x) + tf.square(gradient_y))
        gradient_direction = tf.atan2(gradient_y, gradient_x)

        # apply non-maximum suppression to thin out the edges 
        suppressed = self.non_maximum_suppression(gradient_magnitude, gradient_direction)

        # filter weak and strong edges using thresholds for hysteresis
        strong_edges = tf.where(suppressed >= self.high, 1.0, 0.0)
        weak_edges = tf.where(tf.logical_and(suppressed >= self.low, suppressed < self.high), 1.0, 0.0)

        edges = self.hysteresis_thresholding(strong_edges, weak_edges)

        return edges 
    
    # Non-maximum suppression method.
    def non_maximum_suppression(self, gradient_magnitude, gradient_direction):
        # Convert gradient direction to degrees and normalize.
        angle = gradient_direction * (180.0 / tf.constant(3.14159))
        angle = tf.where(angle < 0, angle + 180, angle)

        # Helper function to shift tensor.
        def shifted_tensor(tensor, dx, dy):
            return tf.roll(tensor, shift=[dx, dy], axis=[1, 2])

        # Define neighbors for comparison.
        neighbours = {
            0: (shifted_tensor(gradient_magnitude, 0, -1), shifted_tensor(gradient_magnitude, 0, 1)),
            45: (shifted_tensor(gradient_magnitude, -1, 1), shifted_tensor(gradient_magnitude, 1, -1)),
            90: (shifted_tensor(gradient_magnitude, -1, 0), shifted_tensor(gradient_magnitude, 1, 0)),
            135: (shifted_tensor(gradient_magnitude, -1, -1), shifted_tensor(gradient_magnitude, 1, 1))
        }

        # Apply non-maximum suppression based on angle.
        suppressed = tf.where(
            # Angle close to 0°
            (tf.logical_or(
                tf.less_equal(angle, 22.5), tf.greater(angle, 157.5)
            ) & 
            (gradient_magnitude >= neighbours[0][0]) & 
            (gradient_magnitude >= neighbours[0][1])) |
            
            # Angle close to 45°
            (tf.logical_and(
                tf.greater(angle, 22.5), tf.less_equal(angle, 67.5)
            ) & 
            (gradient_magnitude >= neighbours[45][0]) & 
            (gradient_magnitude >= neighbours[45][1])) |
            
            # Angle close to 90°
            (tf.logical_and(
                tf.greater(angle, 67.5), tf.less_equal(angle, 112.5)
            ) & 
            (gradient_magnitude >= neighbours[90][0]) & 
            (gradient_magnitude >= neighbours[90][1])) |
            
            # Angle close to 135°
            (tf.logical_and(
                tf.greater(angle, 112.5), tf.less_equal(angle, 157.5)
            ) & 
            (gradient_magnitude >= neighbours[135][0]) & 
            (gradient_magnitude >= neighbours[135][1])),

            # Keep pixels if the magnitude is greater than or equal to its neighbors.
            gradient_magnitude, tf.zeros_like(gradient_magnitude)
        )

        return suppressed  
        
    def hysteresis_thresholding(self, strong_edges, weak_edges):
        # Initialize edges to strong_edges and previous_edges to all zeros
        edges = tf.identity(strong_edges)
        previous_edges = tf.zeros_like(edges)

        # Define condition and body for the while loop
        def condition(edges, previous_edges):
            # Loop until edges stop changing (i.e., when edges equal previous_edges)
            return tf.reduce_any(tf.not_equal(edges, previous_edges))
        
        def body(edges, previous_edges):
            previous_edges = edges
            
            # Cast to bool for logical operation
            weak_edges_bool = tf.cast(weak_edges, tf.bool)
            strong_edges_connected_bool = tf.cast(self.connected_to_strong(edges), tf.bool)

            # Logical AND to find weak edges connected to strong edges
            updated_edges = tf.where(
                tf.logical_and(weak_edges_bool, strong_edges_connected_bool),
                1.0,  # Replace with 1.0 for connected edges
                edges  # Keep original edges for others
            )
            return updated_edges, previous_edges

        # Run the while loop to update edges
        edges, _ = tf.while_loop(condition, body, [edges, previous_edges])
        
        return edges
    
    # check if weak edges are connected to strong edges in the neighbourhood. Used in hysteresis_thresholding

    def connected_to_strong(self, edges):
        def shifted_tensor(tensor, dx, dy):
            return tf.roll(tensor, shift=[dx, dy], axis=[1, 2])
        
    # Sum of neighbors around each pixel
        neighbor_sum = (
            shifted_tensor(edges, 0, 1) + shifted_tensor(edges, 0, -1) +
            shifted_tensor(edges, 1, 0) + shifted_tensor(edges, -1, 0) +
            shifted_tensor(edges, 1, 1) + shifted_tensor(edges, -1, -1) +
            shifted_tensor(edges, 1, -1) + shifted_tensor(edges, -1, 1)
        )
        return neighbor_sum > 0
    
    # return the configuration for saving and loading
    def get_configuration(self):
        config = super(CannyEdgeLayer, self).get_config()
        config.update({'low_threshold': self.low,
                       'high_threshold': self.high})
        return config

