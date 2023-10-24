import cv2
import numpy as np


def get_image_gradients(image, ksize=3, sigma=2, smooth=True, abs=False):
    if smooth:
    # smooth
        image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
        image = cv2.medianBlur(image, ksize)
    # sobel
    sobelx = cv2.Sobel(src=image, ddepth=cv2.CV_64F,
                        dx=1, dy=0, ksize=ksize)
    sobely = cv2.Sobel(src=image, ddepth=cv2.CV_64F,
                        dx=0, dy=1, ksize=ksize)
    # abs
    magnitude = np.sqrt((sobelx.copy() ** 2) + (sobely.copy() ** 2))
    orientation = np.rad2deg(np.arctan(sobely.copy(), sobelx.copy()))
    if abs:
        sobelx_abs = cv2.convertScaleAbs(sobelx.copy())
        sobely_abs = cv2.convertScaleAbs(sobely.copy())
        return sobelx_abs, sobely_abs, magnitude, orientation
    else:
        return sobelx, sobely, magnitude, orientation


# TODO: change discretization?
def non_max_suppression(image, angle):
    M, N = image.shape
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
                    q = image[y, x-1]
                    r = image[y, x+1]
                # angle 45
                elif (22.5 <= angle[y, x] < 67.5):
                    q = image[y-1, x-1]
                    r = image[y+1, x+1]
                # angle 90
                elif (67.5 <= angle[y, x] < 90) or (-90 < angle[y, x] <= -67.5):
                    q = image[y-1, x]
                    r = image[y+1, x]
                # angle 135
                elif (-67.5 < angle[y, x] <= -22.5):
                    q = image[y+1, x-1]
                    r = image[y-1, x+1]

                if (image[y,x] >= q) and (image[y,x] >= r):
                    Z[y,x] = image[y,x]
                else:
                    Z[y,x] = 0

            except IndexError as e:
                # ayo?
                Z[y,x] = 0
                # pass
    return Z


def threshold(image, highThreshold, lowThreshold, weak, strong):

    highThreshold = image.max() * highThreshold
    lowThreshold = highThreshold * lowThreshold

    M, N = image.shape
    res = np.zeros((M,N), dtype=np.int32)

    strong_i, strong_j = np.where(image >= highThreshold)
    zeros_i, zeros_j = np.where(image < lowThreshold)

    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res)

def hysteresis(image, weak, strong):

    M, N = image.shape

    for y in range(1, M-1):
        for x in range(1, N-1):
            if (image[y,x] == weak):
                try:
                    if ((image[y+1, x-1] == strong) or (image[y+1, x] == strong) or (image[y+1, x+1] == strong)
                        or (image[y, x-1] == strong) or (image[y, x+1] == strong)
                        or (image[y-1, x-1] == strong) or (image[y-1, x] == strong) or (image[y-1, x+1] == strong)):
                        image[y, x] = strong
                    else:
                        image[y, x] = 0
                except IndexError as e:
                    pass
    return image
