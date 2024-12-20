""" Statistical Digital Signal Processing """

import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import cv2
import numpy as np
from mse import *



#sigma values: 0; 0.010; 1; 2 [0.000,0.0010,1.000,2.000]
TEST_SIGMA = [0.000,0.00001,0.001,0.100,1.000,10]

ROWS = 2
COLUMNS = len(TEST_SIGMA)

wiener_MSE1 = np.zeros(len(TEST_SIGMA))
inverse_MSE1 = np.zeros(len(TEST_SIGMA))
wiener_MSE2 = np.zeros(len(TEST_SIGMA))
inverse_MSE2= np.zeros(len(TEST_SIGMA))

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

def add_gaussian_noise(image, temp_sigma=TEST_SIGMA, mean=0):
    """Function to add Gaussian noise to an image"""
    # Generate Gaussian noise
    noise = np.random.normal(mean, temp_sigma, image.shape)

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
plt.figure(figsize=(2*len(TEST_SIGMA), len(TEST_SIGMA)))

# # Show Image 1
# plt.subplot(ROWS, COLUMNS, 1)
# plt.imshow(I1, cmap='gray')
# plt.title('Image 1')
# plt.axis('off')

# # Show  Image 2
# plt.subplot(ROWS, COLUMNS, 6)
# plt.imshow(I2, cmap='gray')
# plt.title('Image 2')
# plt.axis('off')

# take the 2D Fast Fourier Transform of both the images
I1_fft = scipy.fft.fft2(I1)
I2_fft = scipy.fft.fft2(I2)

# this method scales the filter to image size before creating the blur
I1_blur_filter_big = create_filter(len(I1), len(I1) // 50 + 1, 5)
I2_blur_filter_big = create_filter(len(I2), len(I2) // 50 + 1, 5)

#plt.imshow(I1_blur_filter_big, cmap="gray")

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

# # Show the blurred images
# plt.subplot(ROWS, COLUMNS, 2)
# plt.imshow(I1_blurred_image, cmap='gray')
# plt.title('Blurred Image 1')
# plt.axis('off')

# plt.subplot(ROWS, COLUMNS, 7)
# plt.imshow(I2_blurred_image, cmap='gray')
# plt.title('Blurred Image 2')
# plt.axis('off')

# Adding Gaussian noise to both images
i = 0
for sigma in TEST_SIGMA:
    I1_blurred_noisy = add_gaussian_noise(I1_blurred_image,sigma)
    I2_blurred_noisy = add_gaussian_noise(I2_blurred_image,sigma)
    # Calculate the perfect value for K for each individual image using K = noise_variance^2/((1/n)*summation()):
    K1 = (sigma**2)*I1.size / np.sum(np.abs(I1)**2)
    K2 = (sigma**2)*I2.size / np.sum(np.abs(I2)**2)
    print(sigma)
    # Show the blurred images
    # plt.subplot(ROWS, COLUMNS, 3)
    # plt.imshow(I1_blurred_noisy.image, cmap='gray')
    # plt.title('Blurred Image 1 with noise')
    # plt.axis('off')

    # plt.subplot(ROWS, COLUMNS, 8)
    # plt.imshow(I2_blurred_noisy.image, cmap='gray')
    # plt.title('Blurred Image 2 with noise')
    # plt.axis('off')

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
    #plt.subplot(ROWS, COLUMNS, 1+i)
    #plt.imshow(I1_blurred_noisy_inverse, cmap='gray')
    #plt.title('Image 1, $\sigma=$'+str(sigma))
    #plt.axis('off')

    #plt.subplot(ROWS, COLUMNS, 5+i)
    #plt.imshow(I2_blurred_noisy_inverse, cmap='gray')
    #plt.title('Image 2, $\sigma=$'+str(sigma))
    #plt.axis('off')

    # Calculate the MSE of the inverse using mse.py
    I1_mse_inv = calculate_mse(I1, I1_blurred_noisy_inverse)
    I2_mse_inv = calculate_mse(I2, I2_blurred_noisy_inverse)
    #print("inverse:")
    #print(I1_mse)
    #print(I2_mse)

    # Wiener Filter:
    wiener_filter_1 = np.conj(I1_fft_filter_big)/(np.abs(I1_fft_filter_big)**2 + (K1-0*K1))
    wiener_filter_2 = np.conj(I2_fft_filter_big)/(np.abs(I2_fft_filter_big)**2 + (K2-0*K2))

    #Calculate the cleaned images:
    I1_blurred_noisy_wiener = np.abs(scipy.fft.ifft2(I1_blurred_noisy_fft * wiener_filter_1))
    I2_blurred_noisy_wiener = np.abs(scipy.fft.ifft2(I2_blurred_noisy_fft * wiener_filter_2))

    # Calculate the MSE of the inverse using mse.py
    I1_mse = calculate_mse(I1, I1_blurred_noisy_wiener)
    I2_mse = calculate_mse(I2, I2_blurred_noisy_wiener)
    #print("Wiener:")
    #print(I1_mse)
    #print(I2_mse)
    
    # Show all the images
    plt.subplot(ROWS,COLUMNS,1+i)
    plt.imshow(I1_blurred_noisy_wiener, cmap='gray')
    plt.title('Image 1, $\sigma=$'+str(sigma))
    plt.axis('off')

    plt.subplot(ROWS,COLUMNS,len(TEST_SIGMA)+1+i)
    plt.imshow(I2_blurred_noisy_wiener, cmap='gray')
    plt.title('Image 2, $\sigma=$'+str(sigma))
    plt.axis('off')
    wiener_MSE1[i] = round(I1_mse,3)
    wiener_MSE2[i] = round(I2_mse,3)
    inverse_MSE1[i] = round(I1_mse_inv,3)
    inverse_MSE2[i] = round(I2_mse_inv,3)
    i += 1


plt.savefig("blur_filter_wiener.svg", format="svg")
plt.show()
print("wiener1")
print(wiener_MSE1)
print("wiener2")
print(wiener_MSE2)
print("inverse1")
print(inverse_MSE1)
print("inverse2")
print(inverse_MSE2)
#plt.subplot(1,2,1)
#plt.imshow(I1_blurred_noisy_inverse, cmap='gray')
#plt.title('Blurred Image 1 inverse filtered without noise')
#plt.axis('off')

#plt.subplot(1,2,2)
#plt.imshow(I2_blurred_noisy_inverse, cmap='gray')
#plt.title('Blurred Image 2 inverse filtered without noise')
#plt.axis('off')


