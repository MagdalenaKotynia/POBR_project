from utils import read_image, median_filter
from thresholding import region_growing, get_segments,crop_segment
from geometric_features import compute_features, compute_reference_metrics
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from utils import median_filter, max_filter


def detect(image_path):
    detected_segment_coords = []
    detected_conditions = []
    detected_segments = []
    img = read_image(image_path, max_size=600)
    #img = median_filter(img, filter_size=5)
    segmented_img, seeds = region_growing(img, threshold= 9, neighbourhood=4) #T = 7 dobre dla o1
    print("Wyznaczono punkty startowe do segmentacji")
    segmented_img = median_filter(segmented_img, filter_size=3)

    cropped_segments, segments = get_segments(segmented_img)
    print("Wyznaczono segmenty")
    reference_main = compute_reference_metrics(
        'C:/Users/magda/OneDrive/Dokumenty/Informatyka mgr/3 sem/POBR/POBR_projekt/obrazy/references/', option='main')
    reference_small = compute_reference_metrics(
        'C:/Users/magda/OneDrive/Dokumenty/Informatyka mgr/3 sem/POBR/POBR_projekt/obrazy/references/', option='small')
    reference_whole = compute_reference_metrics(
        'C:/Users/magda/OneDrive/Dokumenty/Informatyka mgr/3 sem/POBR/POBR_projekt/obrazy/references/', option='whole')
    for cropped_segment, segment in zip(cropped_segments, segments):
        #cropped_segment[0] = max_filter(cropped_segment[0], filter_size=3)
        #cv2.imshow("segment", segment[0])
        segment_coords, condition, detected_segment = check_partial_conditions(cropped_segment, reference_main, reference_small)
        if segment_coords is not None:
            detected_segment_coords.append(segment_coords)
            detected_conditions.append(condition)
            detected_segments.append(segment[0])
    detected_seg_data = zip(detected_segment_coords, detected_conditions, detected_segments)
    print("oceniono wszystkie segmenty")
    apple_segments_coords, apple_conditions, apple_segments = check_whole_conditions(detected_seg_data, reference_whole)

    draw_bboxes(img, detected_segment_coords, option='parts')
    if apple_segments is not None:
        draw_bboxes(img, apple_segments_coords, option='whole')
    #merge_segments(segments[4], segments[5])
    cv2.imshow("segmented_img", segmented_img)
    cv2.waitKey(0)


def draw_bboxes(img, segments_coords, option='parts'):
    if option == 'parts':
        color = 'r'
    if option == 'whole':
        color = 'g'
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for seg_coords in segments_coords:
        x = seg_coords[2]
        y = seg_coords[3]
        height = seg_coords[1] - y
        width = seg_coords[0] - x
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    plt.show()


def check_partial_conditions(segment, ref_main, ref_small):
    features = compute_features(segment[0])
    # small_min_condition = features > ref_small[2]
    # small_max_condition = features < ref_small[1]
    # main_min_condition = features > ref_main[2]
    # main_max_condition = features < ref_main[1]
    # small_conditions = np.min(np.concatenate((small_min_condition, small_max_condition)).reshape((2, 5)), axis=0)
    # main_conditions = np.min(np.concatenate((main_min_condition, main_max_condition)).reshape((2, 5)), axis=0)
    #cv2.imshow("segment", segment[0])
    small_condition = np.abs(features - ref_small[0]) < ref_small[3]
    main_condition = np.abs(features - ref_main[0]) < ref_main[3]

    # if np.count_nonzero(main_conditions) > 3:
    #     condition = 'main'
    #     segment_coords = segment[1]
    #     return segment_coords, condition, segment[0]
    # if np.count_nonzero(small_conditions) > 3:
    #     condition = 'small'
    #     segment_coords = segment[1]
    #     return segment_coords, condition, segment[0]
    if np.count_nonzero(main_condition) > 3:
        cond = 'main'
        segment_coords = segment[1]
        return segment_coords, cond, segment[0]
    if np.count_nonzero(small_condition) > 3:
        cond = 'small'
        segment_coords = segment[1]
        return segment_coords, cond, segment[0]
    else:
        return None, None, None


def check_whole_conditions(seg_data, whole_refs):
    seg_data = list(seg_data)
    detected_segments_coords = []
    detected_conditions = []
    detected_segments = []
    for i in range(0, len(seg_data)):
        seg_1 = seg_data[i]
        if seg_1[1] == 'main':
            for j in range(0, len(seg_data)):
                seg_2 = seg_data[j]
                if seg_2[1] == 'small':
                    new_segment, new_segment_coords = merge_segments(seg_1, seg_2)
                    #cv2.imshow("seg", new_segment)
                    features = compute_features(new_segment)
                    # min_condition = features > whole_refs[2]
                    # max_condition = features < whole_refs[1]
                    # whole_conditions = np.min(np.concatenate((min_condition, max_condition)).reshape((2, 5)), axis=0)
                    whole_condition = np.abs(features - whole_refs[0]) < whole_refs[3]
                    if np.count_nonzero(whole_condition) > 3:
                        condition = 'whole'
                        detected_segments_coords.append(new_segment_coords)
                        detected_conditions.append(condition)
                        detected_segments.append(new_segment)

    return detected_segments_coords, detected_conditions, detected_segments


def merge_segments(seg_1, seg_2):
    seg_coords_1 = seg_1[0]
    seg_coords_2 = seg_2[0]
    seg_data_1 = seg_1[2]
    seg_data_2 = seg_2[2]
    x_max = max(seg_coords_1[0], seg_coords_2[0])
    y_max = max(seg_coords_1[1], seg_coords_2[1])
    x_min = min(seg_coords_1[2], seg_coords_2[2])
    y_min = min(seg_coords_1[3], seg_coords_2[3])
    merged_seg_coords = [x_max, y_max, x_min, y_min]
    merged_segments = np.zeros(seg_data_2.shape)
    merged_segments[seg_data_1 == 255] = 255
    merged_segments[seg_data_2 == 255] = 255
    merged_cropped_seg, segment_coords = crop_segment(merged_segments)
    #merged_cropped_seg = max_filter(merged_cropped_seg, filter_size=3)
    #cv2.imshow("segment", merged_cropped_seg)
    return merged_cropped_seg, merged_seg_coords



detect('obrazy\o2.jpg')
