import cv2
import sympy as sp
import numpy as np


# GLOBALS #
# Write down the Desired SNR
snr_value = 6 # dB
image_path = './shape256.jpg'


# Load the image
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
# Define the variables
A, sigma, SNR = sp.symbols('A sigma SNR')
# Create the expression
expr = 20 * sp.log((A / sigma), 10)
# Define the equation
equation = sp.Eq(SNR, expr)
print(f'The SNR equation is: {equation}')
# Calculate A, The Dynamic Range of the Image
# dynamic_range = np.max(image) - np.min(image)
dynamic_range = np.std(image)
print(f'Image Sigma is: {dynamic_range}')
# Substitute the values for SNR and A
equation_with_values = equation.subs({SNR: snr_value, A: dynamic_range})
# Solve for sigma
solutions = sp.solve(equation_with_values, sigma)
# Convert the solution to an actual number
sigma_value = eval(str(solutions[0]))
print(f'Sigma Value is: {sigma_value}')

# Check SNR again
equation_with_values = equation.subs({sigma: sigma_value, A: dynamic_range})
# Solve for sigma
solutions = sp.solve(equation_with_values, SNR)
# Convert the solution to an actual number
snr_value = eval(str(solutions[0]))
print(f'SNR Value is: {snr_value}')

# Generate Gaussian noise with zero mean and the calculated variance
height, width = image.shape
noise = np.random.normal(0, sigma_value, (height, width))
# Add the noise to the image
noisy_image = image + noise
# Clip the pixel values to ensure they are within the valid range [0, 255]
noisy_image = np.clip(noisy_image, 0, 255)
# Convert the noisy_image back to uint8 data type (if not already)
noisy_image = noisy_image.astype(np.uint8)
# Save the noisy image
cv2.imwrite('noisy_image.jpg', noisy_image)
# # show the image
# cv2.imshow('noisy image', noisy_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
