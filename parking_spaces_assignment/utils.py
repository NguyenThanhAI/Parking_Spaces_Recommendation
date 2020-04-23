import numpy as np


def find_unique_values_and_frequency(cropped_mask, id, use_unified_id=True):
    assert len(cropped_mask.shape) == 3 and cropped_mask.shape[2] == 2, "Cropped mask must have rank 3 and axis 2 must be 2"

    cropped_mask = np.reshape(cropped_mask, newshape=(-1, 2))
    unique_values, counts = np.unique(cropped_mask, return_counts=True, axis=0)
    print(unique_values, counts)
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