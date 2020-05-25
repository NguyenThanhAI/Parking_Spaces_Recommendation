import os
import time
from pathlib import Path
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN


class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1
    DETECTION_MIN_CONFIDENCE = 0.5


def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [1]:
            car_boxes.append(box)

    return np.array(car_boxes)

ROOT_DIR = Path(".")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_cars_and_vehicles_0008.h5")

if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_DIR = os.path.join(ROOT_DIR, "images")

RESULT_DIR = os.path.join(ROOT_DIR, "results")

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR, exist_ok=True)

model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

model.load_weights(COCO_MODEL_PATH, by_name=True)

image_path = "201911_20191129_2_2019-11-29_09-00-00_38343.jpg"

image_path = os.path.join(IMAGE_DIR, image_path)

img = cv2.imread(image_path)

rgb_img = img[:, :, ::-1]

start = time.time()
results = model.detect([rgb_img], verbose=0)
end = time.time()

result = results[0]

rois, scores, class_ids, masks = result["rois"], result["scores"], result["class_ids"], result["masks"]

masks = masks.transpose(2, 0, 1)

color = np.array([255, 255, 0], dtype=np.uint8)

color_mask = np.zeros_like(img, dtype=np.uint8)

for roi, score, class_id, mask in zip(rois, scores, class_ids, masks):
    if score >= 0.0 and class_id in [1]:
        color = np.array([np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)], dtype=np.uint8)
        y_min, x_min, y_max, x_max = roi
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        color_mask = np.where(mask[:, :, np.newaxis], color[np.newaxis, np.newaxis, :], color_mask)

#cv2.imshow("", img)
#cv2.waitKey(0)
#
#cv2.imshow("", color_mask)
#cv2.waitKey(0)

img = np.where(color_mask > 0, cv2.addWeighted(img, 0.3, color_mask, 0.7, 0), img)

#cv2.imshow("", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

print(round(end - start, 3))

cv2.imwrite(os.path.join(RESULT_DIR, os.path.basename(image_path)), img)
