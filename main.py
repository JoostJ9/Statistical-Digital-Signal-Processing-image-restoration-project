"""Module"""

import matplotlib.pyplot as plt
import scipy.io
import cv2
import numpy as np
# Load the .mat file
mat_data = scipy.io.loadmat('img_restoration.mat')
# Print the contents
print(mat_data)

# Extract the images
I1 = mat_data['I1']
I2 = mat_data['I2']
print(I2)
# Plotting the images
plt.figure(figsize=(10, 5))        #create a figure that does not cover the whole screen.

# Show image 1
plt.subplot(1, 2, 1)
plt.imshow(I1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

# Show image 2
plt.subplot(1, 2, 2)
plt.imshow(I2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

# Display the images
plt.show()

# Try fft2
I1_fft = scipy.fft.fft2(I1)
I2_fft = scipy.fft.fft2(I2)

# Create filter
def create_filter(size,radius,type):
    # Create matrix with zeros
    filter = np.zeros((size,size), dtype=np.float32)
    # Create a center point
    center = (size // 2, size // 2)
    filter = cv2.circle(filter, center=center, radius=radius, color=(255,), thickness=-1)
    # Normalize to make an average filter (to normalize to length 1 and ensure the brightness of image stays the same)
    print(filter)
    plt.subplot(1, 2, 1)
    plt.imshow(filter, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')

    # Display the images
    filter /= filter/np.sum(filter)
    print(filter)
    plt.subplot(1, 2, 2)
    plt.imshow(filter, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')

    # Display the images
    plt.show()
    return filter
create_filter(21,8,5)