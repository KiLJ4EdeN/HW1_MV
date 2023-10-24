import cv2
import numpy as np


img = cv2.imread('./noisy_image.jpg', cv2.IMREAD_UNCHANGED)
img = cv2.GaussianBlur(img, (3, 3), 2)
img = cv2.medianBlur(img, 3)

sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F,
                    dx=1, dy=0, ksize=3)
sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F,
                    dx=0, dy=1, ksize=3)
# abs
magnitude = np.sqrt((sobelx ** 2) + (sobely ** 2))
sobelx_abs = cv2.convertScaleAbs(sobelx)
sobely_abs = cv2.convertScaleAbs(sobely)
cv2.imwrite('sobelx.jpg', sobelx_abs)
cv2.imwrite('sobely.jpg', sobely_abs)
cv2.imwrite('magnitude.jpg', magnitude)
