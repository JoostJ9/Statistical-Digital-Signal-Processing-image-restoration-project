"""
function made for the Statistical Digital Signal processing image restoration project
by Thomas Prins and Joost Jaspers for TU Delft
to calculate the mean square error between 2 images.
"""

import numpy as np

def calculate_mse(original_image, restored_image):
    """Function to calculate Mean Square Error between two images."""
    # Ensure both images are in the same data type and shape
    original_image = original_image.astype(np.float64)
    restored_image = restored_image.astype(np.float64)

    # Compute the Mean Square Error
    mse = np.mean((original_image - restored_image) ** 2)
    
    return mse
