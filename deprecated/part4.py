import cv2
import numpy as np

def non_max_suppression(img, angle):
    M, N = img.shape
    # zeros of image size
    Z = np.zeros((M,N), dtype=np.float32)

    # loop over image
    for y in range(1,M-1):
        for x in range(1,N-1):
            try:
                q = 255
                r = 255

                # angle 0
                if (-22.5 < angle[y, x] < 22.5):
                    q = img[y, x-1]
                    r = img[y, x+1]
                # angle 45
                elif (22.5 <= angle[y, x] < 67.5):
                    q = img[y-1, x-1]
                    r = img[y+1, x+1]
                # angle 90
                elif (67.5 <= angle[y, x] < 90) or (-90 < angle[y, x] <= -67.5):
                    q = img[y-1, x]
                    r = img[y+1, x]
                # angle 135
                elif (-67.5 < angle[y, x] <= -22.5):
                    q = img[y+1, x-1]
                    r = img[y-1, x+1]

                if (img[y,x] >= q) and (img[y,x] >= r):
                    Z[y,x] = img[y,x]
                else:
                    Z[y,x] = 0

            except IndexError as e:
                # ayo?
                Z[y,x] = 0
                # pass
    return Z

def threshold(img, highThreshold, lowThreshold, weak, strong):

    highThreshold = img.max() * highThreshold
    lowThreshold = highThreshold * lowThreshold

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res)

def hysteresis(img, weak, strong):

    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img
    
img = cv2.imread('./noisy_image.jpg', cv2.IMREAD_UNCHANGED)
img = cv2.GaussianBlur(img, (3, 3), 2)
img = cv2.medianBlur(img, 3)
sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F,
                    dx=1, dy=0, ksize=3)
sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F,
                    dx=0, dy=1, ksize=3)
# compute the gradient magnitude and orientation
magnitude = np.sqrt((sobelx ** 2) + (sobely ** 2))
orientation = np.rad2deg(np.arctan(sobely, sobelx))
# see some values
print(magnitude[35:40, 35:40])
print(orientation[35:40, 35:40])
nms_mag = non_max_suppression(magnitude, orientation)
print(nms_mag[35:40, 35:40])
# finally threshold
# somewhat a little bit better
nms_th = threshold(nms_mag, 0.40, 0.90, 75, 255)
nms_th = hysteresis(nms_th, 75, 255)
# ret, nms_th2 = cv2.threshold(nms_mag, 200, 255, cv2.THRESH_BINARY)

# print(nms_th == nms_th2)
cv2.imwrite('magnitude.jpg', magnitude)
cv2.imwrite('orientation.jpg', orientation)
cv2.imwrite('magnitude_nms.jpg', nms_mag)
cv2.imwrite('magnitude_th.jpg', nms_th)
# cv2.imwrite('magnitude_th2.jpg', nms_th2)
