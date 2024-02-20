import cv2
import numpy as np


def decim_im(image, N):
    """
    Resizes the input image by a factor of 1/2^(N-1) using linear interpolation and returns the resized image as a float32 array.
    Parameters:
    - image: the input image to be resized
    - N: the decimation factor
    Returns:
    - im_decimee: the resized image as a float32 array
    """

    im_decimee = cv2.resize(image, None, fx=1/2**(N-1), fy=1/2**(N-1), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return im_decimee