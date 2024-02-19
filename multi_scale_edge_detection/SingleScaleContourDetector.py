from filters.gradient_magnitude import gradient_magnitude
from SSCD.SNMS import seuil_hyst, supp_non_max
from inhibition.contour_inhibition import contour_inhibition


def SingleScaleContourDetector_with_inhibition(image , sigma , alpha):
    # Calcul du Gradient avec un Noyau Gaussien
    M_sigma = gradient_magnitude(image, sigma)
    # Afficher le résultat final
   
    c_sigma = contour_inhibition(M_sigma, sigma, alpha)
    

    ## normalise image 
    
    c_sigma_normal = (c_sigma- c_sigma.min()) / (c_sigma.max() - c_sigma.min())
   
    
    SSCD = seuil_hyst(c_sigma_normal)
    
    return SSCD


def SingleScaleContourDetector_without_inhibition(image , sigma , alpha):
    # Calcul du Gradient avec un Noyau Gaussien
    c_sigma = gradient_magnitude(image, sigma)
    
    c_sigma_normal = (c_sigma- c_sigma.min()) / (c_sigma.max() - c_sigma.min())
    # Seuillage hystérésique et suppression des non-maxima
    # Afficher le résultat final
   
    
 
    
    SSCD = seuil_hyst(c_sigma_normal)
    
    return SSCD