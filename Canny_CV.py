import cv2
import numpy as np

img = cv2.imread('noisy_image.jpg', cv2.IMREAD_UNCHANGED)
# filter first, as this does not happen in opencv with respect to matlab for some reason
# img = cv2.blur(img, (3, 3), 2)
img = cv2.GaussianBlur(img, (3, 3), 2)
img = cv2.medianBlur(img, 3)
cv2.imwrite('filtered.jpg', img)
v = np.median(img)
lower = int(max(0, (1.0 - 0.33) * v))
upper = int(min(255, (1.0 + 0.33) * v))
edge = cv2.Canny(img, 150, 250)
# closing
# edge = cv2.dilate(edge, None, iterations=1)
# edge = cv2.erode(edge, None, iterations=1)
cv2.imwrite('edges.jpg', edge)
