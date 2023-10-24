import numpy as np
import sympy as sp


def add_gaussian_noise(image, sigma):
    # Generate Gaussian noise with zero mean and the calculated variance
    height, width = image.shape
    noise = np.random.normal(0, sigma, (height, width))
    # Add the noise to the image
    noisy_image = image + noise
    # Clip the pixel values to ensure they are within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    # Convert the noisy_image back to uint8 data type (if not already)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


class ImageSNR(object):
    def __init__(self) -> None:
        # Define the variables
        self.A, self.sigma, self.SNR = sp.symbols('A sigma SNR')
        # Create the expression
        self.expr = 20 * sp.log((self.A / self.sigma), 10)
        # Define the equation
        self.equation = sp.Eq(self.SNR, self.expr)
        print(f'The SNR equation is: {self.equation}')

    def get_snr(self, image, sigma_value):
        equation_with_values = self.equation.subs({self.sigma: sigma_value, self.A: np.std(image)})
        # Solve for sigma
        solutions = sp.solve(equation_with_values, self.SNR)
        # Convert the solution to an actual number
        snr_value = eval(str(solutions[0]))
        return snr_value
    
    def get_sigma(self, image, snr_value):
        equation_with_values = self.equation.subs({self.SNR: snr_value, self.A: np.std(image)})
        # Solve for sigma
        solutions = sp.solve(equation_with_values, self.sigma)
        # Convert the solution to an actual number
        sigma_value = eval(str(solutions[0]))
        return sigma_value
