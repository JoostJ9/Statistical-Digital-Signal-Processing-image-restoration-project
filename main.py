import matplotlib.pyplot as plt
import scipy.io
import cv2
import numpy as np
# Load the .mat file
mat_data = scipy.io.loadmat('img_restoration.mat')

# Print the contents
#print(mat_data)

# Extract the images
I1 = mat_data['I1']
print(I1.shape)
I2 = mat_data['I2']

#print(I2)

# Plotting the images
plt.figure(figsize=(10, 5))

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
#plt.show()

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
    filter /= np.sum(filter)
    return filter



# this method scales the filter to image size before creating the blur
blur_filter_big = create_filter(len(I1),len(I1)//50 ,5)
blur_filter_big = scipy.fft.fftshift(blur_filter_big)
fft_filter_big = scipy.fft.fft2(blur_filter_big)
#print(fft_blur_big)
# this method pads zeroes to create an equally sized blur filter array compared to the image
blur_filter_small = create_filter(80,10,5)
padded_blur_filter = np.zeros_like(I1, dtype = np.float32)
fh, fw = blur_filter_small.shape
padded_blur_filter[:fh, :fw] = blur_filter_small

#Show the filters
plt.subplot(1, 2, 1)
plt.imshow(blur_filter_big, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(padded_blur_filter, cmap='gray')
plt.title('Image 2')
plt.axis('off')

# Display the images
plt.show()

fft_filter_small = scipy.fft.fft2(padded_blur_filter)

blurred_fft_small = fft_filter_small * I1_fft
blurred_fft_big = fft_filter_big* I1_fft

blurred_image_small = np.abs(scipy.fft.ifft2(blurred_fft_small))
blurred_image_big = np.abs(scipy.fft.ifft2(blurred_fft_big))

#Show the filters
plt.subplot(1, 3, 1)
plt.imshow(blurred_image_small, cmap='gray')
plt.title('Image small')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blurred_image_big, cmap='gray')
plt.title('Image big')
plt.axis('off')

# Display the images
plt.show()