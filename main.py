"""
Image restoration project
For statistical digital signal processing, MSc electrical engineering
Thomas Prins(Student nummer) and Joost Jaspers(5372925)
"""

import matplotlib.pyplot as plt
import scipy.io
import cv2
import numpy as np

def create_filter(size,radius,type):
    """Function to create a filter"""
    # Create matrix with zeros
    image_filter = np.zeros((size,size), dtype=np.float32)

    # Create a center point
    center = (size // 2, size // 2)
    image_filter = cv2.circle(image_filter,
                              center=center,
                              radius=radius,
                              color=(255,),
                              thickness=-1)

    # Normalize to make an average filter
    # (to normalize to length 1 and ensure the brightness of image stays the same)
    image_filter /= np.sum(image_filter)

    return image_filter

# Load the .mat file
mat_data = scipy.io.loadmat('img_restoration.mat')

# Extract the images from the mat file into uint8 matrices. 
I1 = mat_data['I1']
I2 = mat_data['I2']

# Creating a figure in matplotlib
plt.figure(figsize=(12, 6))

# Create image 1 from matrix I1 and put it in a subplot of the figure.
plt.subplot(2, 3, 1)
plt.imshow(I1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

# Create image 2 from matrix I2 and put it in a subplot of the figure.
plt.subplot(2, 3, 4)
plt.imshow(I2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

# Display the images
#plt.show()

# take the 2D Fast Fourier Transform of both the images. 
I1_fft = scipy.fft.fft2(I1)
I2_fft = scipy.fft.fft2(I2)

# this method scales the filter to image size before creating the blur
I1_blur_filter_big = create_filter(len(I1),len(I1)//50 ,5)
I2_blur_filter_big = create_filter(len(I2),len(I2)//50 ,5)

#Show the filters
plt.subplot(2, 3, 2)
plt.imshow(I1_blur_filter_big, cmap='gray')
plt.title('Filter for image 1')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(I2_blur_filter_big, cmap='gray')
plt.title('Filter for image 2')
plt.axis('off')

# FFT-shift the filter.
I1_blur_filter_big = scipy.fft.fftshift(I1_blur_filter_big)
I2_blur_filter_big = scipy.fft.fftshift(I2_blur_filter_big)

# Take the 2D fast fourtier transform of the filter.
I1_fft_filter_big = scipy.fft.fft2(I1_blur_filter_big)
I2_fft_filter_big = scipy.fft.fft2(I2_blur_filter_big)

# this method pads zeroes to create an equally sized blur filter array compared to the image
# blur_filter_small = create_filter(80,10,5)
# padded_blur_filter = np.zeros_like(I1, dtype = np.float32)
# fh, fw = blur_filter_small.shape
# padded_blur_filter[:fh, :fw] = blur_filter_small

# Display the images
#plt.show()

#fft_filter_small = scipy.fft.fft2(padded_blur_filter)

#blurred_fft_small = fft_filter_small * I1_fft
I1_blurred_fft = I1_fft_filter_big * I1_fft
I2_blurred_fft = I2_fft_filter_big * I2_fft

#blurred_image_small = np.abs(scipy.fft.ifft2(blurred_fft_small))

#Take the inverse 2D fast fourier transform of the convolution of the filter and the image.
I1_blurred_image = np.abs(scipy.fft.ifft2(I1_blurred_fft))
I2_blurred_image = np.abs(scipy.fft.ifft2(I2_blurred_fft))

#Show the filters
plt.subplot(2, 3, 3)
plt.imshow(I1_blurred_image, cmap='gray')
plt.title('Blurred image 1')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(I2_blurred_image, cmap='gray')
plt.title('Blurred image 2')
plt.axis('off')

# Display the images
plt.show()
