from utils import read_image, median_filter
from thresholding import region_growing, get_segments
from geometric_features import compute_features, compute_reference_metrics
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from utils import median_filter, max_filter


def detect(image_path):
    detected_segment_coords = []
    img = read_image(image_path, max_size=600)
    #img = median_filter(img, filter_size=5)
    segmented_img, seeds = region_growing(img, threshold= 9, neighbourhood=4) #T = 7 dobre dla o1
    #segmented_img = median_filter(segmented_img, filter_size=3)
    cv2.imshow("segmented_img", segmented_img)
    segments = get_segments(segmented_img)
    reference_main = compute_reference_metrics(
        'C:/Users/magda/OneDrive/Dokumenty/Informatyka mgr/3 sem/POBR/POBR_projekt/obrazy/references/', option='main')
    reference_small = compute_reference_metrics(
        'C:/Users/magda/OneDrive/Dokumenty/Informatyka mgr/3 sem/POBR/POBR_projekt/obrazy/references/', option='small')
    for segment in segments:
        segment[0] = max_filter(segment[0], filter_size=3)
        #cv2.imshow("segment", segment[0])
        features = compute_features(segment[0])
        small_min_condition = features > reference_small[2]
        small_max_condition = features < reference_small[1]
        main_min_condition = features > reference_main[2]
        main_max_condition = features < reference_main[1]
        small_conditions = np.min(np.concatenate((small_min_condition, small_max_condition)).reshape((2, 5)), axis=0)
        main_conditions = np.min(np.concatenate((main_min_condition, main_max_condition)).reshape((2, 5)), axis=0)

        if np.count_nonzero(small_conditions) > 3:
            segment_coords= segment[1]
            detected_segment_coords.append(segment_coords)
        if np.count_nonzero(main_conditions) > 3:
            segment_coords= segment[1]
            detected_segment_coords.append(segment_coords)
    draw_bboxes(img, detected_segment_coords)


def draw_bboxes(img, segments_coords):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for seg_coords in segments_coords:
        x = seg_coords[2]
        y = seg_coords[3]
        height = seg_coords[1] - y
        width = seg_coords[0] - x
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


detect('obrazy\o5.jpg')
