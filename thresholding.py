import numpy as np
from utils import max_filter, min_filter, histogram, rgb2grey


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def get_seeds(img):
    neighbourhood_size = 50
    threshold = 120
    data_max = max_filter(img, neighbourhood_size)
    maxima = (img == data_max)
    data_min = min_filter(img, neighbourhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    seeds = maxima.nonzero()
    seeds = np.asarray(seeds)
    seeds = np.transpose(seeds)
    print("Wyznaczono punkty startowe do segmentacji.")
    return seeds


def region_growing(image, threshold, neighbourhood=8):
    grey_img = rgb2grey(image)
    seeds = get_seeds(grey_img)
    num_seeds = seeds.shape[0]
    if num_seeds < 100:
        seg_colour_diff = int(155//num_seeds)
        new_seg_colour = 100
    else:
        seg_colour_diff = 1
        new_seg_colour = 255 - num_seeds

    binarized_img = np.zeros(image[:, :, 0].shape)
    neighbours = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    height, width = image[:, :, 0].shape

    for i in range(num_seeds):  #
        seed = seeds[i]
        q = Queue()
        q.enqueue(seed)

        while q.size() > 0:
            p = q.dequeue()
            binarized_img[p[0], p[1]] = new_seg_colour
            for j in range(neighbourhood):
                x_new = p[0] + neighbours[j][0]
                y_new = p[1] + neighbours[j][1]
                is_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)
                if is_inside:
                    if binarized_img[x_new, y_new] == 0:
                        R = int(image[p[0], p[1], 0])
                        G = int(image[p[0], p[1], 1])
                        B = int(image[p[0], p[1], 2])
                        R_new = int(image[x_new, y_new, 0])
                        G_new = int(image[x_new, y_new, 1])
                        B_new = int(image[x_new, y_new, 2])
                        distance = ((R_new - R) ** 2 + (G_new - G) ** 2 + (B_new - B) ** 2) ** 0.5
                        if distance < threshold:
                            q.enqueue((x_new, y_new))
                            binarized_img[x_new, y_new] = new_seg_colour
        new_seg_colour = new_seg_colour + seg_colour_diff
    return binarized_img, seeds


def get_segments(rg_img):
    segments_ids = np.unique(rg_img)[1:]
    cropped_segments_list = []
    segments_list = []
    for id in segments_ids:
        cropped_segment_data = []
        segment_data = []
        segment = np.zeros(rg_img.shape)

        segment[rg_img != id] = 0
        segment[rg_img == id] = 255
        if np.count_nonzero(segment == 255) < 100:
            pass
        else:
            cropped_segment, segment_coords = crop_segment(segment)
            cropped_segment_data.append(cropped_segment)
            cropped_segment_data.append(segment_coords)
            cropped_segments_list.append(cropped_segment_data)
            segment_data.append(segment)
            segment_data.append(segment_coords)
            segments_list.append(segment_data)
    print("Wyznaczono segmenty.")
    return cropped_segments_list, segments_list


def crop_segment(segment, space=0.1):
    segment_pixels = (segment == 255).nonzero()
    x_max = np.max(segment_pixels[1])
    y_max = np.max(segment_pixels[0])
    x_min = np.min(segment_pixels[1])
    y_min = np.min(segment_pixels[0])
    segment_coords = [x_max, y_max, x_min, y_min]
    x_diff = x_max - x_min
    y_diff = y_max - y_min

    max_size = max(x_diff, y_diff)
    segment_size = max_size + 2 * max_size*space

    if max_size == y_diff:
        cropped_segment = segment[int(max(y_min - max_size*space, 0)):int(min(y_max + max_size*space, segment.shape[0])), \
                        int(max(x_min - (segment_size - x_diff)/2, 0)): int(min(x_max + (segment_size - x_diff)/2, segment.shape[1]))]
    if max_size == x_diff:
        cropped_segment = segment[int(max(y_min - (segment_size - y_diff)/2, 0)):int(min(y_max + (segment_size - y_diff)/2, segment.shape[0])), \
                        int(max(x_min - max_size*space, 0)): int(min(x_max + max_size*space, segment.shape[1]))]
    return cropped_segment, segment_coords


def otsu(image):
    hist = histogram(image)
    hist_sum = sum(hist.values())
    between_class_variances = {}
    for i, key in enumerate(hist.keys()):

        background = {key: value for key, value in hist.items() if key < i}

        foreground = {key: value for key, value in hist.items() if key >= i}

        W_b = sum(background.values())/hist_sum
        W_f = sum(foreground.values())/hist_sum
        try:
            mi_b = sum(key * value for key, value in background.items())/sum(background.values())
        except ZeroDivisionError:
            mi_b = 0
        try:
            mi_f = sum(key * value for key, value in foreground.items())/sum(foreground.values())
        except ZeroDivisionError:
            mi_f = 0
        between_class_variance = W_b*W_f*(mi_b - mi_f)**2
        between_class_variances[key] = between_class_variance
    threshold = max(between_class_variances, key=between_class_variances.get)

    return threshold

