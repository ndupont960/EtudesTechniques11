
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt



def gradient_magnitude(gray_image, sigma=1.0):
    # Apply Gaussian smoothing
    blurred_image = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma)

    # Compute the x and y gradients using the Sobel operators
    gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    gradient_magnitude = sqrt(gradient_x**2 + gradient_y**2)

    # Normalize to [0, 1]
    gradient_magnitude /= np.max(gradient_magnitude)

    return gradient_magnitude





 

