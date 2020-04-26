import numpy as np
import cv2


def find_unique_values_and_frequency(cropped_mask, id, use_unified_id=True):
    assert len(cropped_mask.shape) == 3 and cropped_mask.shape[2] == 2, "Cropped mask must have rank 3 and axis 2 must be 2"

    cropped_mask = np.reshape(cropped_mask, newshape=(-1, 2))
    unique_values, counts = np.unique(cropped_mask, return_counts=True, axis=0)
    #print (unique_values, counts)
    results = {}
    for i, unique_value in enumerate(unique_values):
        if np.less(unique_value, 0).any():
            continue
        else:
            if use_unified_id:
                check_id = unique_value[0]
            else: # Use vehicle_id
                check_id = unique_value[1]
            if check_id != id:
                continue
            else:
                results[tuple(unique_value)] = counts[i]
    #print(results)
    return results


def write_information(frame, num_vehicles, num_available_ps):
    height, width = frame.shape[:2]
    cv2.putText(img=frame, text="Number of vehicles: {}".format(num_vehicles), org=(int(0.05 * width), int(0.05 * height)),
                fontFace=cv2.QT_FONT_NORMAL, fontScale=0.5, color=(255, 255, 0), thickness=1)
    cv2.putText(img=frame, text="Number of available parking spaces: {}".format(num_available_ps), org=(int(0.05 * width), int(0.1 * height)),
                fontFace=cv2.QT_FONT_NORMAL, fontScale=0.5, color=(255, 255, 0), thickness=1)