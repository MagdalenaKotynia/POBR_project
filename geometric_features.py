import math
import os
import numpy as np
from utils import read_image,binarize
from thresholding import crop_segment


def indices_mat(image):
    y_len = image.shape[0]
    x_len = image.shape[1]
    x = np.arange(x_len)
    x = x.reshape(1, x_len)
    y = np.arange(y_len)
    y = y.reshape(y_len, 1)
    return x, y


def m(image, p, q):
    if image.ndim == 3:
        img = np.copy(image[:, :, 0])
    else:
        img = np.copy(image)
    img[img != 255] = 1
    img[img == 255] = 0
    x, y = indices_mat(img)

    m = sum(sum(np.multiply(np.matmul(y ** q, x ** p), img)))
    return m.astype('int64')


def M(image, p, q):
    if image.ndim == 3:
        img = np.copy(image[:, :, 0])
    else:
        img = np.copy(image)
    img[img != 255] = 1
    img[img == 255] = 0

    x, y = indices_mat(img)
    i = m(image, 1, 0) / m(image, 0, 0)
    j = m(image, 0, 1) / m(image, 0, 0)

    xp = ((x - m(image, 1, 0) / m(image, 0, 0))) ** p
    yp = ((y - m(image, 0, 1) / m(image, 0, 0))) ** q


    M = sum(sum(np.multiply(np.matmul(yp, xp), img)))
    return M


def compute_features(image):
    M1 = (M(image, 2, 0) + M(image, 0, 2)) / m(image, 0, 0) ** 2

    # M2 = ((M(image, 2, 0) - M(image, 0, 2)) ** 2 + 4 * M(image, 1, 1) ** 2 )/ m(image, 0, 0) ** 4
    #
    # M3 = ((M(image, 3,0) - 3 * M(image, 1, 2))**2 + (3*M(image, 2, 1) - M(image, 0, 3))**2) / m(image, 0, 0)**5
    #
    # M4 = ((M(image, 3, 0) + M(image, 1, 2))**2 + (M(image, 2, 1) + M(image, 0, 3))**2) / m(image, 0, 0)**5
    #
    # M5 = ((M(image, 3, 0) - 3 * M(image, 1, 2)) * (M(image, 3, 0) + M(image, 1, 2)) *\
    #       ((M(image, 3, 0)+ M(image, 1, 2))**2 - 3 * (M(image, 2, 1) + M(image, 0, 3))**2) +\
    #     (3 * M(image, 2, 1) - M(image, 0, 3)) * (M(image, 2, 1) + M(image, 0, 3)) *\
    #     (3 * (M(image, 3, 0) + M(image, 1, 2))**2 - (M(image, 2, 1) + M(image, 0, 3))**2)) / m(image, 0, 0) ** 10
    #
    # M6 = ((M(image, 2, 0) - M(image, 0, 2))*((M(image, 3, 0) + M(image, 1, 2))**2 - (M(image, 2, 1) + M(image, 0, 3))**2) +\
    #              4 * M(image, 1, 1) * (M(image, 3, 0) + M(image, 1, 2)) * (M(image, 2, 1) + M(image, 0, 3))) / m(image, 0, 0)**7\

    M7 = (M(image, 2, 0) * M(image, 0, 2) - M(image, 1, 1)**2) / m(image, 0, 0)**4

    # M8 = (M(image, 3, 0) * M(image, 1, 2) +  M(image, 2, 1) * M(image, 0, 3) - M(image, 1, 2)**2 - M(image, 2, 1)**2) / \
    #      M(image, 0, 0)**5
    #
    # M9 = (M(image, 2, 0) * (M(image, 2, 1) * M(image, 0, 3) - M(image, 1, 2)**2) +\
    #              M(image, 0, 2) * (M(image, 0, 3) * M(image, 1, 2) - M(image, 2, 1)**2) -\
    #              M(image, 1, 1) * (M(image, 3, 0) * M(image, 0, 3) - M(image, 2, 1) * M(image, 1, 2))) / m(image, 0, 0)**7\
    #
    # M10 = ((M(image, 3, 0) * M(image, 0, 3) - M(image, 1, 2) * M(image, 2, 1))**2 -\
    #               4*(M(image, 3, 0)*M(image, 1, 2) - M(image, 2, 1)**2)*(M(image, 0, 3) * M(image, 2, 1) - M(image, 1, 2))) / m(image, 0, 0)**10

    w3 = W3(image)

    w9 = W9(image)


    #features = [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, w3, w9]
    features = [M1, M7, w3, w9]
    return features


def img_center(image):
    y = image.shape[0] // 2
    x = image.shape[1] // 2
    return y, x


def angle(img_center, obj_center):
    radians = math.atan2(img_center[0] - obj_center[0], img_center[1] - obj_center[1])
    degrees = math.degrees(radians)
    return degrees


def area(image):
    obj_pixels = image != 255
    if image.ndim == 3:
        area = obj_pixels.sum() / 3
    else:
        area = obj_pixels.sum()
    return area


def circuit(image):
    if image.ndim == 3:
        img = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3
    else:
        img = np.copy(image)
    circuit = 0
    for y in range(0, img.shape[0] - 1):
        for x in range(0, img.shape[1] - 1):
            if (img[y, x] != img[y, x + 1]):
                circuit = circuit + 1
            if (img[y, x] != img[y + 1, x]):
                circuit = circuit + 1
    return circuit


def W3(image):
    L = circuit(image)
    S = area(image)
    w3 = L / (2 * math.sqrt(math.pi * S)) - 1
    return w3


def W9(image):
    L = circuit(image)
    S = area(image)
    w9 = 2 * math.sqrt(math.pi*S) / L
    return w9


def central_i(image):
    return math.floor(m(image, 1, 0) / m(image, 0, 0))


def central_j(image):
    return math.floor(m(image, 0, 1) / m(image, 0, 0))


def compute_reference_metrics(img_dir, option):
    reference_features = np.empty(0)
    if option == 'main':
        option = '_main'
    files = os.listdir(img_dir)
    for file in files:
        if file[-9:-4] == option:
            img = read_image(img_dir + file)
            img = binarize(img, threshold=80)

            cropped_img, segment_coords = crop_segment(img)
            img_features = compute_features(cropped_img)
            reference_features = np.append(reference_features, img_features, axis=0)
    reference_features = np.reshape(reference_features, (int(len(reference_features)/len(img_features)), len(img_features)))
    mean_vals = np.mean(reference_features, axis=0)
    max_vals = np.max(reference_features, axis=0)
    min_vals = np.min(reference_features, axis=0)
    std_vals = np.std(reference_features, axis=0)

    results = [mean_vals, max_vals, min_vals, std_vals]

    return results

#





