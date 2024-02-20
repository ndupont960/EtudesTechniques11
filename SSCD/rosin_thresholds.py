import numpy as np
from scipy.ndimage import gaussian_filter1d
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt

def doubleRosinThr(originalImage, precission=0.01, gaussianSigma=1):

    """
    Function to perform double thresholding on the input image.

    Args:
        originalImage: The input image as a numpy array.
        precission: The precision used for thresholding (default is 0.01).
        gaussianSigma: The standard deviation for Gaussian smoothing of the histogram (default is 1).

    Returns:
        doubleThr: Numpy array containing the low and high thresholds for the image.
    """
    # Validate Arguments
   
    assert (precission > 0), 'Error: Precision must be a positive value.'

    binsNumber = round(1 / precission)
    assert (binsNumber > 10), 'Error: Wrong (too coarse) precision.'

    assert (gaussianSigma >= 0), 'Error: GaussianSigma must be a non-negative value.'

    # Preprocessing
    originalImage = originalImage.astype(np.float64)
    if np.max(originalImage) > 1.001:
        factor = 255
        originalImage = originalImage / 255
    else:
        factor = 1

 

    # Histogram Processing
    histogram, _ = np.histogram(originalImage.flatten(), bins=binsNumber, range=[0, 1])
    if gaussianSigma == float('inf'):
        # print(histogram.shape)
        poss, _ = find_peaks(np.concatenate(([0], histogram)))
        while len(poss) > 1:
            histogram = gaussian_filter(histogram, sigma=1.0)
            # print(histogram.shape)
            poss, _ = find_peaks(np.concatenate(([0], histogram)))
    elif gaussianSigma > 0:
        histogram = gaussian_filter(histogram, sigma=gaussianSigma)

    # Finding the Peak
    histPeakVal, _ = find_peaks(histogram)
    if histPeakVal.size == 0:
        histPeak = np.argmax(histogram)
    elif len(histPeakVal) == 1:
        histPeak = histPeakVal[0]
        if histogram[histPeak] < 0.10 * histogram[0]:
            histPeak = np.argmax(histogram)
    elif len(histPeakVal) > 1:
        peak_vals = histogram[histPeakVal]
        max_val = peak_vals.max()
        histPeak = histPeakVal[peak_vals == max_val][0]
        if np.sum(histogram[histPeak:]) < 0.10 * np.sum(histogram) or histogram[histPeak] < 0.33 * histogram.max():
            histPeak = np.argmax(histogram)

    # Finding the Max
    histEnd = binsNumber - 1
    while histogram[histEnd] == 0 and histogram[histEnd - 1] == 0 and histEnd > 1:
        histEnd -= 1

    if histEnd == 1:
        raise ValueError('Error: The histogram was empty or the image was 0-valued.')

    # Processing: Finding the high THR
    highThr = histPeak + getMaxDist(histogram[histPeak:histEnd]) - 1

    # Processing: Finding the low THR
    lowThr = histPeak + getMaxDist(histogram[histPeak:highThr]) - 1



    ## plot the treshold on the histogramm plot 
    # Assuming histogram is the input histogram and highThr, lowThr are the threshold values
    # plt.plot(histogram, color='gray')
    # plt.axvline(x=highThr, color='r', linestyle='--', label='High Threshold')
    # plt.axvline(x=lowThr, color='g', linestyle='--', label='Low Threshold')
    # plt.legend()
    # plt.show()

    
        # Output formatting
    perc = np.array([lowThr, highThr]) / binsNumber
    doubleThr = perc * factor

    return doubleThr


def getMaxDist(funcValues):
    """
    Calculate the furthest point from a line defined by two points in the given function values array.

    Parameters:
    funcValues (array): An array of function values.

    Returns:
    int: The index of the furthest point from the line.
    """
    x = 0
    y = len(funcValues) - 1

    hx = funcValues[x]
    hy = funcValues[y]

    # Checking if the denominator is zero to avoid division by zero
    if y - x == 0:
        return x

    # Defining the line
    a = (hy - hx) / (y - x)
    b = hx - (x * a)

    distances = np.zeros_like(funcValues)
    maxDistance = -1
    furthersPos = x

    for z in range(x, y + 1):
        hz = funcValues[z]
        dist = np.abs(a * z - hz + b) / np.sqrt(a ** 2 + 1)
        distances[z] = dist
        if dist > maxDistance:
            maxDistance = dist
            furthersPos = z

    return furthersPos
