import cv2
import numpy as np
import matplotlib.pyplot as plt
from multi_scale_processing.decimation import decim_im

from multi_scale_processing.dto import doubling_and_ticking_operator

from multi_scale_edge_detection.SingleScaleContourDetector import SingleScaleContourDetector_with_inhibition , SingleScaleContourDetector_without_inhibition
 
def multi_scale_contour_detection_with_inhibition(image_normalized, N=4, alpha=0.005):
    # Tableau de matrices et calcul des images à plusieurs échelles
    images_decimees = [None] * N
    images_decimees[0] = image_normalized

    for n in range(2, N + 1):
        images_decimees[n-1] = decim_im(image_normalized, n)  # Assuming you have the function decim_im defined

    # Détection de contours à chaque échelle
    contours_binaires = [None] * N

    for n in range(N):
        contours_binaires[n] = SingleScaleContourDetector_with_inhibition(images_decimees[n], 1, alpha)  # Assuming you have the function SingleScaleContourDetector defined

    # Combine les contours à différentes échelles : de la plus grossière et en descendant progressivement
    resultat_final = contours_binaires[N-1]

    for n in range(N-2, -1, -1):
        # Dilatation du résultat actuel
        dto = doubling_and_ticking_operator(contours_binaires[n+1])  # Assuming you have the function doubling_and_ticking_operator defined

        # Convert b1 to the same data type as dto
        contours_binaires[n] = contours_binaires[n].astype(np.uint8)

        # Combinaison avec la carte de contours à l'échelle suivante
        resultat_final = cv2.bitwise_and(contours_binaires[n], dto)
        ##resultat_final = np.where(contours_binaires[n] == 0, dto, contours_binaires[n])
        plt.imshow(resultat_final, cmap='gray')
        plt.title('Résultat de la détection de contours multiscale avec inhibition du contour, étape : ' + str(n+1))
        plt.show()

    plt.imshow(resultat_final, cmap='gray')
    plt.title('Résultat de la détection de contours multiscale avec inhibition du contour')
    plt.show()

    return resultat_final



def multi_scale_contour_detection_without_inhibition(image_normalized, N=4, alpha=0.005):
    # Tableau de matrices et calcul des images à plusieurs échelles
    images_decimees = [None] * N
    images_decimees[0] = image_normalized

    for n in range(2, N + 1):
        images_decimees[n-1] = decim_im(image_normalized, n)  
    # Détection de contours à chaque échelle
    contours_binaires = [None] * N

    for n in range(N):
        contours_binaires[n] = SingleScaleContourDetector_without_inhibition(images_decimees[n], 1, alpha)  # Assuming you have the function SingleScaleContourDetector defined

    # Combine les contours à différentes échelles : de la plus grossière et en descendant progressivement
    resultat_final = contours_binaires[N-1]

    for n in range(N-2, -1, -1):
        # Dilatation du résultat actuel
        dto = doubling_and_ticking_operator(contours_binaires[n+1])  # Assuming you have the function doubling_and_ticking_operator defined

        # Convert b1 to the same data type as dto
        contours_binaires[n] = contours_binaires[n].astype(np.uint8)

        # Combinaison avec la carte de contours à l'échelle suivante
        resultat_final = cv2.bitwise_and(contours_binaires[n], dto)
        ##resultat_final = np.where(contours_binaires[n] == 0, dto, contours_binaires[n])
        plt.imshow(resultat_final, cmap='gray')
        plt.title('Résultat de la détection de contours multiscale sans inhibition du contour, étape : ' + str(n+1))
        plt.show()

    plt.imshow(resultat_final, cmap='gray')
    plt.title('Résultat de la détection de contours multiscale sans inhibition du contour')
    plt.show()

    return resultat_final