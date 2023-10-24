import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_hist(img, save_path):
    fig = plt.figure(figsize=(6, 6))
    hist, bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc='upper left')
    fig.savefig(save_path)

img = cv2.imread('shape256.jpg', cv2.IMREAD_UNCHANGED)
print(img.shape)
print(np.min(img))
print(np.max(img))
print(len(np.unique(img)))
print(np.unique(img))

# optional, eq hist
equalized_img = cv2.equalizeHist(img)

# plot hists for both
plot_hist(img, 'hist.png')
plot_hist(equalized_img, 'hist_eq.png')

cv2.imwrite('output.jpg', img)
cv2.imwrite('output_eq.jpg', equalized_img)
