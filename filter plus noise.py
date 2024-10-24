""" Statistical Digital Signal Processing """

import matplotlib.pyplot as plt
import scipy.io
import cv2
import numpy as np

def create_filter(size, radius, type):
    """Function to create a filter"""
    # Create matrix with zeros
    image_filter = np.zeros((size, size), dtype=np.float32)

    # Create a center point
    center = (size // 2, size // 2)
    image_filter = cv2.circle(image_filter,
                              center=center,
                              radius=radius,
                              color=(255,),
                              thickness=-1)

    # Normalize to make an average filter
    image_filter /= np.sum(image_filter)

    return image_filter

def add_gaussian_noise(image, mean=0, sigma=10):
    """Function to add Gaussian noise to an image"""
    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma, image.shape)

    # Add the noise to the original image
    noisy_image = image + noise

    # Clip values to be in the valid range [0, 255] for uint8 images
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image.astype(np.uint8)

# Load the .mat file
mat_data = scipy.io.loadmat('img_restoration.mat')

# Extract the images from the mat file into uint8 matrices. 
I1 = mat_data['I1']
I2 = mat_data['I2']

# Creating a figure in matplotlib
plt.figure(figsize=(12, 6))

# # Show Image 1
# plt.subplot(1, 2, 1)
# plt.imshow(I1, cmap='gray')
# plt.title('Image 1')
# plt.axis('off')

# # Show  Image 2
# plt.subplot(1, 2, 2)
# plt.imshow(I2, cmap='gray')
# plt.title('Image 2')
# plt.axis('off')

# plt.savefig("plot.svg", format="svg")

# Show Image 1
plt.subplot(2, 4, 1)
plt.imshow(I1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

# Show  Image 2
plt.subplot(2, 4, 5)
plt.imshow(I2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

# take the 2D Fast Fourier Transform of both the images
I1_fft = scipy.fft.fft2(I1)
I2_fft = scipy.fft.fft2(I2)

# this method scales the filter to image size before creating the blur
I1_blur_filter_big = create_filter(len(I1), len(I1) // 50, 5)
I2_blur_filter_big = create_filter(len(I2), len(I2) // 50, 5)

# Show the filters
plt.subplot(2, 4, 2)
plt.imshow(I1_blur_filter_big, cmap='gray')
plt.title('Filter for Image 1')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(I2_blur_filter_big, cmap='gray')
plt.title('Filter for Image 2')
plt.axis('off')

# FFT-shift the filter.
I1_blur_filter_big = scipy.fft.fftshift(I1_blur_filter_big)
I2_blur_filter_big = scipy.fft.fftshift(I2_blur_filter_big)

# Take the 2D FFT of the filter.
I1_fft_filter_big = scipy.fft.fft2(I1_blur_filter_big)
I2_fft_filter_big = scipy.fft.fft2(I2_blur_filter_big)

# Multiply the FFTs of the images and filters in the frequency domain.
I1_blurred_fft = I1_fft_filter_big * I1_fft
I2_blurred_fft = I2_fft_filter_big * I2_fft

# Take the inverse FFT to get the blurred images.
I1_blurred_image = np.abs(scipy.fft.ifft2(I1_blurred_fft))
I2_blurred_image = np.abs(scipy.fft.ifft2(I2_blurred_fft))

# Show the blurred images
plt.subplot(2, 4, 3)
plt.imshow(I1_blurred_image, cmap='gray')
plt.title('Blurred Image 1')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(I2_blurred_image, cmap='gray')
plt.title('Blurred Image 2')
plt.axis('off')

# Adding Gaussian noise to both images
I1_blurred_noisy = add_gaussian_noise(I1_blurred_image)
I2_blurred_noisy = add_gaussian_noise(I2_blurred_image)

# Show the blurred images
plt.subplot(2, 4, 4)
plt.imshow(I1_blurred_noisy, cmap='gray')
plt.title('Blurred Image 1 with noise')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(I2_blurred_noisy, cmap='gray')
plt.title('Blurred Image 2 with noise')
plt.axis('off')

plt.savefig("plot.svg", format="svg")

# Display the images
plt.show()
