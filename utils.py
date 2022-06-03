import numpy as np
from matplotlib import pyplot as plt
import cv2


def read_image(img_path, max_size = 800):
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    bigger_size = np.max([h, w])
    if bigger_size > max_size:
        if bigger_size == h:
            new_w = max_size
            new_h = int(w/(h/max_size))
        if bigger_size == w:
            new_h = max_size
            new_w = int(h/(w/max_size))
        img = cv2.resize(img, (new_h, new_w), interpolation = cv2.INTER_AREA)
    return img


def rgb2grey(image):
    #for rgb images
    img = np.copy(image)
    grey = img[:, :, 0] * 0.2989 //1 + img[:, :, 1] * 0.5870 //1 + img[:, :, 2] * 0.1140 //1
    print(grey)
    return grey.astype(np.uint8)


def bgr2grey(image):
    #for bgr images
    img = np.copy(image)
    grey = img[:, :, 2] * 0.2989 //1 + img[:, :, 1] * 0.5870 //1 + img[:, :, 0] * 0.1140 //1
    print(grey)
    return grey.astype(np.uint8)


def bgr2rgb(image):
    img = np.copy(image)
    R_tmp = img[:, :, 2]
    img[:,:, 2] = img[:, :, 0]
    img[:, :, 0] = R_tmp
    return img


def binarize(image, threshold):
    binary = np.copy(image)
    binary[binary < threshold] = 0
    binary[binary >= threshold] = 255
    return binary


def histogram(image, display=False):
    # sorted = np.sort(image, axis=None)
    unique, counts = np.unique(image, return_counts=True)
    hist_dict = dict(zip(unique, counts))
    if display:
        plt.bar(range(len(hist_dict)), counts, tick_label=None)
        plt.show()
    return hist_dict


def augment(image, aug_size):
    img = np.copy(image)
    left = img[:, 0:aug_size // 2]
    right = img[:, -aug_size // 2:-1]
    aug_image = np.concatenate((left, img, right), axis=1)
    up = aug_image[0:aug_size // 2]
    down = aug_image[-aug_size // 2:-1]
    aug_image = np.concatenate((up, aug_image, down), axis=0)
    return aug_image


def get_window(image, pix_row_idx, pix_col_idx, filter_size):
    if image.ndim == 3:
        window = image[(pix_row_idx - filter_size // 2):(pix_row_idx + filter_size // 2 + 1),
                 (pix_col_idx - filter_size // 2):(pix_col_idx + filter_size // 2) + 1, :]
    else:
        window = image[(pix_row_idx - filter_size // 2):(pix_row_idx + filter_size // 2),
                 (pix_col_idx - filter_size // 2):(pix_col_idx + filter_size // 2)]
    return window


def median_filter(image, filter_size=7):
    aug_image = augment(image, filter_size)
    median_image = np.copy(aug_image)
    for row_idx in range(filter_size // 2, aug_image.shape[0] - filter_size // 2):
        for col_idx in range(filter_size // 2, aug_image.shape[1] - filter_size // 2):
            window = get_window(aug_image, row_idx, col_idx, filter_size)
            if image.ndim == 3:
                window_greyscale = window[:, :, 0] / 3 + window[:, :, 1] / 3 + window[:, :, 2] / 3

                sorted = np.sort(window_greyscale, axis=None)
                median = sorted[filter_size ^ 2 // 2]
                median_row_idx = np.where(window_greyscale == median)[0][0]

                median_col_idx = np.where(window_greyscale == median)[1][0]
                median_image[row_idx, col_idx, :] = window[median_row_idx, median_col_idx, :]
            else:
                window_greyscale = np.copy(window)

                sorted = np.sort(window_greyscale, axis=None)
                median = sorted[filter_size ^ 2 // 2]
                median_row_idx = np.where(window_greyscale == median)[0][0]

                median_col_idx = np.where(window_greyscale == median)[1][0]
                median_image[row_idx, col_idx] = window[median_row_idx, median_col_idx]
    if image.ndim == 3:
        median_image = median_image[filter_size//2: -filter_size//2+1, filter_size//2: -filter_size//2+1, :]
    else:
        median_image = median_image[filter_size // 2: -filter_size // 2 + 1, filter_size // 2: -filter_size // 2 + 1]
    return median_image


def max_filter(image, filter_size=7):
    aug_image = augment(image, filter_size)
    max_image = np.copy(aug_image)
    for row_idx in range(filter_size // 2, aug_image.shape[0] - filter_size // 2):
        for col_idx in range(filter_size // 2, aug_image.shape[1] - filter_size // 2):
            window = get_window(aug_image, row_idx, col_idx, filter_size)
            if image.ndim == 3:
                window_greyscale = window[:, :, 0] / 3 + window[:, :, 1] / 3 + window[:, :, 2] / 3
                max = np.max(window_greyscale)
                max_row_idx = np.where(window_greyscale == max)[0][0]

                max_col_idx = np.where(window_greyscale == max)[1][0]
                max_image[row_idx, col_idx, :] = window[max_row_idx, max_col_idx, :]
            else:
                window_greyscale = np.copy(window)
                max = np.max(window_greyscale)
                max_row_idx = np.where(window_greyscale == max)[0][0]

                max_col_idx = np.where(window_greyscale == max)[1][0]
                max_image[row_idx, col_idx] = window[max_row_idx, max_col_idx]

    if image.ndim == 3:
        max_image = max_image[filter_size//2: -filter_size//2+1, filter_size//2: -filter_size//2+1, :]
    else:
        max_image = max_image[filter_size // 2: -filter_size // 2 + 1, filter_size // 2: -filter_size // 2 + 1]
    return max_image


def min_filter(image, filter_size=7):
    aug_image = augment(image, filter_size)
    min_image = np.copy(aug_image)
    for row_idx in range(filter_size // 2, aug_image.shape[0] - filter_size // 2):
        for col_idx in range(filter_size // 2, aug_image.shape[1] - filter_size // 2):
            window = get_window(aug_image, row_idx, col_idx, filter_size)
            if image.ndim == 3:
                window_greyscale = window[:, :, 0] / 3 + window[:, :, 1] / 3 + window[:, :, 2] / 3
                min = np.min(window_greyscale)
                min_row_idx = np.where(window_greyscale == min)[0][0]

                min_col_idx = np.where(window_greyscale == min)[1][0]
                min_image[row_idx, col_idx, :] = window[min_row_idx, min_col_idx, :]
            else:
                window_greyscale = np.copy(window)
                min = np.min(window_greyscale)
                min_row_idx = np.where(window_greyscale == min)[0][0]

                min_col_idx = np.where(window_greyscale == min)[1][0]
                min_image[row_idx, col_idx] = window[min_row_idx, min_col_idx]
    if image.ndim == 3:
        min_image = min_image[filter_size//2: -filter_size//2+1, filter_size//2: -filter_size//2+1, :]
    else:
        min_image = min_image[filter_size // 2: -filter_size // 2 + 1, filter_size // 2: -filter_size // 2 + 1]
    return min_image

