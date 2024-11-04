import tensorflow as tf
import tensorflow_addons as tfa
from Kernels import sobel_kernel_x, sobel_kernel_y, gaussian_kernel

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
    
# Define a custom layer for canny edge detection. 
class CannyEdgeLayer(tf.keras.layers.Layer):

    def __init__(self, low=0.1, high=0.3, **kwargs):
        super(CannyEdgeLayer, self).__init__(**kwargs)
        self.low = low
        self.high = high

    def build(self, input_shape):
        self.sobel_x = sobel_kernel_x
        self.sobel_y = sobel_kernel_y
        self.sobel_x = tf.reshape(self.sobel_x, [3, 3, 1, 1])
        self.sobel_y = tf.reshape(self.sobel_y, [3, 3, 1, 1])

        gaussian_kernel = gaussian_kernel
        self.gaussian_kernel = tf.reshape(gaussian_kernel, [3, 3, 1, 1])
        super(CannyEdgeLayer, self).build(input_shape)

    def call(self, inputs):
        blurred = tf.nn.conv2d(inputs, self.gaussian_kernel, strides=1, padding='SAME')

        gradient_x = tf.nn.depthwise_conv2d(blurred, self.sobel_x, strides=1, padding='SAME')
        gradient_y = tf.nn.depthwise_conv2d(blurred, self.sobel_y, strides=1, padding ='SAME')

        gradient_magnitude = tf.sqrt(tf.square(gradient_x) + tf.square(gradient_y))
        gradient_direction = tf.atan2(gradient_y, gradient_x)

        suppressed = self.non_maximum_suppression(gradient_magnitude, gradient_direction)

        strong_edges = tf.where(suppressed >= self.high, 1.0, 0.0)
        weak_edges = tf.where((suppressed >= self.low) & (suppressed < self.high, 1.0, 0.0))

        edges = self.hysteresis_thresholding(strong_edges, weak_edges)