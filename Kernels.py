import numpy as np

sobel_kernel_x = np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])

sobel_kernel_y = np.array([[1, 2, 1],
                           [0, 0, 0,],
                           [-1, -2, -1]])

gaussian_kernel = np.array([[1/16, 1/8, 1/16],
                                [1/8, 1/4, 1/8],
                                [1/16, 1/8, 1/16]])
