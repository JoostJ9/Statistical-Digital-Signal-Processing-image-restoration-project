""" Statistical Digital Signal Processing """

import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import cv2
import numpy as np
from mse import *

ROWS = 2
COLUMNS = 5
TEST_SIGMA = 0.005

class test:
    def __init__(self,image,noise): 
        self.image = image
        self.noise = noise

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

def add_gaussian_noise(image, mean=0, sigma=TEST_SIGMA):
    """Function to add Gaussian noise to an image"""
    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma, image.shape)

    # Add the noise to the original image
    noisy_image = image + noise

    # Clip values to be in the valid range [0, 255] for uint8 images
    noisy_image= np.clip(noisy_image, 0, 255)

    return test(noisy_image,noise) #.astype(np.uint8)

# Load the .mat file
mat_data = scipy.io.loadmat('img_restoration.mat')

# Extract the images from the mat file into uint8 matrices.
I1 = mat_data['I1']
I2 = mat_data['I2']

# Creating a figure in matplotlib
plt.figure(figsize=(12, 6))

# Show Image 1
plt.subplot(ROWS, COLUMNS, 1)
plt.imshow(I1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

# Show  Image 2
plt.subplot(ROWS, COLUMNS, 6)
plt.imshow(I2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

# take the 2D Fast Fourier Transform of both the images
I1_fft = scipy.fft.fft2(I1)
I2_fft = scipy.fft.fft2(I2)

# this method scales the filter to image size before creating the blur
I1_blur_filter_big = create_filter(len(I1), len(I1) // 50 + 1, 5)
I2_blur_filter_big = create_filter(len(I2), len(I2) // 50 + 1, 5)

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
plt.subplot(ROWS, COLUMNS, 2)
plt.imshow(I1_blurred_image, cmap='gray')
plt.title('Blurred Image 1')
plt.axis('off')

plt.subplot(ROWS, COLUMNS, 7)
plt.imshow(I2_blurred_image, cmap='gray')
plt.title('Blurred Image 2')
plt.axis('off')

# Adding Gaussian noise to both images
I1_blurred_noisy = add_gaussian_noise(I1_blurred_image)
I2_blurred_noisy = add_gaussian_noise(I2_blurred_image)

# Show the blurred images
plt.subplot(ROWS, COLUMNS, 3)
plt.imshow(I1_blurred_noisy.image, cmap='gray')
plt.title('Blurred Image 1 with noise')
plt.axis('off')

plt.subplot(ROWS, COLUMNS, 8)
plt.imshow(I2_blurred_noisy.image, cmap='gray')
plt.title('Blurred Image 2 with noise')
plt.axis('off')

# plt.savefig("plot.svg", format="svg")

# take the 2D Fast Fourier Transform of both the blurred noisy images
I1_blurred_noisy_fft = scipy.fft.fft2(I1_blurred_noisy.image)
I2_blurred_noisy_fft = scipy.fft.fft2(I2_blurred_noisy.image)

## INVERSE FILTER
epsilon = 0

I1_blurred_noisy_inverse_fft = I1_blurred_noisy_fft/I1_fft_filter_big
I2_blurred_noisy_inverse_fft = I2_blurred_noisy_fft/I2_fft_filter_big

# I1_blurred_noisy_inverse_fft = np.divide(I1_blurred_noisy_fft, I1_fft_filter_big, out=np.zeros_like(I1_blurred_noisy_fft), where=I1_fft_filter_big > 0.0)
# I2_blurred_noisy_inverse_fft = np.divide(I2_blurred_noisy_fft, I2_fft_filter_big, out=np.zeros_like(I2_blurred_noisy_fft), where=I2_fft_filter_big > 0.0)

# Inverse Fast Fourier Transform of both the blurred noisy images
I1_blurred_noisy_inverse = np.abs(scipy.fft.ifft2(I1_blurred_noisy_inverse_fft))
I2_blurred_noisy_inverse = np.abs(scipy.fft.ifft2(I2_blurred_noisy_inverse_fft))

# print(I1_blurred_noisy_inverse)
# print(I2_blurred_noisy_inverse)

# Show the inverse filter images
plt.subplot(ROWS, COLUMNS, 4)
plt.imshow(I1_blurred_noisy_inverse, cmap='gray')
plt.title('Blurred Image 1 inverse')
plt.axis('off')

plt.subplot(ROWS, COLUMNS, 9)
plt.imshow(I2_blurred_noisy_inverse, cmap='gray')
plt.title('Blurred Image 2 inverse')
plt.axis('off')

I1_mse = calculate_mse(I1, I1_blurred_noisy_inverse)
I2_mse = calculate_mse(I2, I2_blurred_noisy_inverse)

print(I1_mse)
print(I2_mse)

K = 0.0001

# Wiener Filter:
wiener_filter_1 = np.conj(I1_fft_filter_big)/(np.abs(I1_fft_filter_big)**2 + K)
wiener_filter_2 = np.conj(I2_fft_filter_big)/(np.abs(I2_fft_filter_big)**2 + K)

#Calculate the cleaned images:
I1_blurred_noisy_wiener = np.abs(scipy.fft.ifft2(I1_blurred_noisy_fft * wiener_filter_1))
I2_blurred_noisy_wiener = np.abs(scipy.fft.ifft2(I2_blurred_noisy_fft * wiener_filter_2))

I1_mse = calculate_mse(I1, I1_blurred_noisy_wiener)
I2_mse = calculate_mse(I2, I2_blurred_noisy_wiener)

print(I1_mse)
print(I2_mse)

# Show all the images
plt.subplot(ROWS,COLUMNS,5)
plt.imshow(I1_blurred_noisy_wiener, cmap='gray')
plt.title('Blurred Image 1 Wiener')
plt.axis('off')

plt.subplot(ROWS,COLUMNS,10)
plt.imshow(I2_blurred_noisy_wiener, cmap='gray')
plt.title('Blurred Image 2 Wiener')
plt.axis('off')

plt.show()

#plt.subplot(1,2,1)
#plt.imshow(I1_blurred_noisy_inverse, cmap='gray')
#plt.title('Blurred Image 1 inverse filtered without noise')
#plt.axis('off')

#plt.subplot(1,2,2)
#plt.imshow(I2_blurred_noisy_inverse, cmap='gray')
#plt.title('Blurred Image 2 inverse filtered without noise')
#plt.axis('off')

#plt.savefig("inverse_without_noise.svg", format="svg")
