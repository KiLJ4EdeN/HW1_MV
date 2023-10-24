import cv2
from utils import print_image_metrics
from snr import ImageSNR, add_gaussian_noise
from canny import threshold, hysteresis, get_image_gradients, non_max_suppression

### GLOBALS ###
IMAGE_PATH = './shape256.jpg'
DESIRED_SNR = 6 # dB
KERNEL_SIZE = 3
SIGMA = 2
SMOOTH = True


def main(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # print some metrics
    print_image_metrics(img)
    # (Q1) show the image
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # SNR
    # (Q2) add noise to the image
    image_snr = ImageSNR()
    noise_sigma = image_snr.get_sigma(img, DESIRED_SNR)
    print(f'Sigma Value is: {noise_sigma}')
    # should be 6dB now
    snr_val = image_snr.get_snr(img, noise_sigma)
    print(f'SNR Value is: {snr_val}')
    # create the noisy image now
    print('Adding Gaussian Noise..')
    noisy_image = add_gaussian_noise(img, noise_sigma)
    cv2.imwrite('noisy_image.jpg', noisy_image)
    # (Q3) show sobel gradients
    print('Getting Sobel Gradients, Magnitude and Orientation..')
    sobelx_abs, sobely_abs, magnitude, orientation = get_image_gradients(noisy_image, ksize=KERNEL_SIZE, sigma=SIGMA, smooth=SMOOTH,
                                                                         abs=True)
    # TODO: change to cv2 imshow outside server
    cv2.imwrite('sobelx.jpg', sobelx_abs)
    cv2.imwrite('sobely.jpg', sobely_abs)
    cv2.imwrite('magnitude.jpg', magnitude)
    # (Q4) nms
    print('Performing NMS..')
    nms_mag = non_max_suppression(magnitude, orientation)
    # DEBUG
    # print(magnitude[35:40, 35:40])
    # print(orientation[35:40, 35:40])
    # print(nms_mag[35:40, 35:40])
    # finally threshold
    # somewhat a little bit better
    print('Double Thresholding..')
    nms_th = threshold(nms_mag, 0.40, 0.90, 75, 255)
    print('Hysteresis as the final step..')
    nms_th = hysteresis(nms_th, 75, 255)
    cv2.imwrite('orientation.jpg', orientation)
    cv2.imwrite('magnitude_nms.jpg', nms_mag)
    cv2.imwrite('magnitude_th.jpg', nms_th)
    # this bad
    # ret, nms_th2 = cv2.threshold(nms_mag, 200, 255, cv2.THRESH_BINARY)

if __name__ == '__main__':
    main(IMAGE_PATH)
