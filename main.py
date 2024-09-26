import matplotlib.pyplot as plt
import scipy.io

# Load the .mat file
mat_data = scipy.io.loadmat('img_restoration.mat')

# Print the contents
print(mat_data)

# Extract the images
I1 = mat_data['I1']
I2 = mat_data['I2']

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
plt.show()
