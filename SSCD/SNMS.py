import numpy as np
from skimage import img_as_float

from SSCD.rosin_thresholds import doubleRosinThr

def get_neighbors(grad_angle, i, j):
    # Determine the neighbors based on the gradient angle

    # Convert angle to degrees
    angle1 = grad_angle * 180.0 / np.pi

    # Make sure the angle is positive
    angle1 = angle1 % 180

    # Quantize the angle to one of four directions: 0, 45, 90, 135 degrees
    quantized_angle = (round(angle1 / 45) * 45) % 180

    # Determine neighboring pixels based on the quantized angle
    if quantized_angle == 0:
        neighbors = [(i, j - 1), (i, j + 1)]
    elif quantized_angle == 45:
        neighbors = [(i - 1, j - 1), (i + 1, j + 1)]
    elif quantized_angle == 90:
        neighbors = [(i - 1, j), (i + 1, j)]
    elif quantized_angle == 135:
        neighbors = [(i - 1, j + 1), (i + 1, j - 1)]
    else:
        raise ValueError("Invalid quantized angle")

    return neighbors[0], neighbors[1]


def supp_non_max(c_sigma):
    # Calcul du gradient
    Fx, Fy = np.gradient(img_as_float(c_sigma))
    angle =  np.pi/2 - np.arctan2(Fy, Fx)
    

    # Suppression des non-maximas
    non_max_suppressed = np.zeros_like(c_sigma)
    for i in range(0, angle.shape[0]-1):
        for j in range(0, angle.shape[1]-1):
            
            if  (angle[i,j] >= 3.14) :
                angle[i,j] = angle[i,j] - np.pi

            


            # Trouver les voisins dans la direction du gradient
            neighb1, neighb2 = get_neighbors(angle[i, j], i, j)
            
            # Suppression des non-maxima
            if c_sigma[i, j] >= c_sigma[neighb1] and c_sigma[i, j] >= c_sigma[neighb2]:
                non_max_suppressed[i, j] = c_sigma[i, j]

    seuils = doubleRosinThr(non_max_suppressed,0.01 , 6)            
    
    return non_max_suppressed , seuils

def seuil_hyst(c_sigma):
    # Suppression des non-maximas
    non_max_suppressed  ,s = supp_non_max(c_sigma)

    seuil_bas = s[0]
    seuil_haut = s[1]
    
    b_sigma = np.zeros_like(c_sigma)
    b_sigma[non_max_suppressed >= seuil_haut] = 1
    
    # Propagation des pixels forts aux faibles
    for i in range(1, c_sigma.shape[0]-1):
        for j in range(1, c_sigma.shape[1]-1):
            if non_max_suppressed[i, j] >= seuil_bas:
                if np.any(b_sigma[i-1:i+2, j-1:j+2] == 1):
                    b_sigma[i, j] = 1
   

    
    return b_sigma 