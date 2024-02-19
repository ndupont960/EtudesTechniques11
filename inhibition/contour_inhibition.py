import cv2
from cvxpy import convolve
import numpy as np
from matplotlib import pyplot as plt


def difference_of_gaussians(sigma):
    """
    A function to calculate the difference of Gaussians given a sigma value.
    """
    # Taille du noyau (un multiple de sigma)
    sz = int(np.ceil(sigma * 3) * 2 + 1)
    
    # Création de la grille de coordonnées
    X, Y = np.meshgrid(np.arange(-sz/2, sz/2 + 1), np.arange(-sz/2, sz/2 + 1))
    
    # Calcul des deux gaussiennes
    G1 = np.exp(-(X**2 + Y**2) / (2 * (sigma * 4)**2))
    G2 = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Difference of Gaussians
    dog = np.maximum(G1 / (2 * np.pi * (sigma * 4)**2) - G2 / (2 * np.pi * sigma**2), 0)
    
    return dog

def normalization_term(sigma):
    """
    Calculate the normalization term for a given sigma value.

    Parameters:
    sigma (float): The standard deviation value for the Gaussian distribution.

    Returns:
    float: The normalization term.
    """
    # Calcul du DoG
    dog = difference_of_gaussians(sigma)
    
    # Calcul du terme de normalisation
    w_sigma = dog / np.sum(dog**2)
    
    return w_sigma

def suppression_term(M_sigma, sigma):
    """
    A function to calculate the suppression term using the DoG, the normalization term, and the convolution with the magnitude of the gradient.
    Parameters:
    - M_sigma: the DoG
    - sigma: the standard deviation
    Returns:
    - t_sigma: the suppression term
    """
    # Calcul du DoG
    w_sigma = normalization_term(sigma)
    
    # Convolution du DoG avec la magnitude du gradient
    t_sigma = convolve(M_sigma, w_sigma, mode='nearest')
    
    return t_sigma

def contour_inhibition(M_sigma, sigma, alpha):
    """
    Calculate the contour inhibition of the input M_sigma using the given sigma and alpha parameters.
    Parameters:
    M_sigma (array): The input array for contour inhibition (gradiant magnitude)
    sigma (float): The sigma parameter
    alpha (float): The alpha parameter
    Returns:
    array: The result of the contour inhibition
    """
    # Calcul du terme de suppression
    t_sigma = suppression_term(M_sigma, sigma)
    
    # Supprimer les valeurs négatives
    t_sigma = np.maximum(t_sigma, 0)
    
    # Inhibition de contour
    c_sigma = M_sigma - alpha * t_sigma
    
    # Supprimer les valeurs négatives
    c_sigma = np.maximum(c_sigma, 0)
    
    return c_sigma