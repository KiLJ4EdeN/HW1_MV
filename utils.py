import numpy as np


def print_image_metrics(image):
    print(f'Image Shape: {image.shape}') # image shape
    print(f'Image Min: {np.min(image)}') # min
    print(f'Image Max: {np.max(image)}') # max
    print(f'Unique Pixel Values: {len(np.unique(image))}') # number of unique pixel values
    # print(np.unique(image)) # unique pixel values
