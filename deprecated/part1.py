import cv2
import numpy as np

img = cv2.imread('shape256.jpg', cv2.IMREAD_UNCHANGED)
print(img.shape) # image shape
print(np.min(img)) # min
print(np.max(img)) # max
print(len(np.unique(img))) # number of unique pixel values
print(np.unique(img)) # unique pixel values

# show the image
cv2.imshow('image', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
