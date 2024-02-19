import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

def doubling_and_ticking_operator(binary_map):
    """
    Apply the Doubling and Ticking Operator (DTO) to a binary map.

    Parameters:
    - binary_map: A 2D numpy array representing the binary map.

    Returns:
    - dto_result: The result of applying the DTO to the binary map.
    """

    # Step 1: Doubling - create a new binary map with doubled coordinates for nonzero pixels
    doubled_map = np.zeros((binary_map.shape[0] * 2, binary_map.shape[1] * 2), dtype=np.uint8)
    doubled_map[::2, ::2] = binary_map

    print(doubled_map)

    # Step 2: Ticking - substitute each nonzero pixel in the doubled map with a disk of radius 3 pixels
    structuring_element = disk(3)  # Create a disk structuring element with radius 3
    dto_result = binary_dilation(doubled_map, structure=structuring_element).astype(np.uint8)

    return dto_result
