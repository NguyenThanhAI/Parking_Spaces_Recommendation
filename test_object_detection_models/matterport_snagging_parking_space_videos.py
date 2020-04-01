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
    NUM_CLASSES = 1 + 80
    DETECTION_MIN_CONFIDENCE = 0.1

ROOT_DIR = Path(".")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

model.load_weights(COCO_MODEL_PATH, by_name=True)

video_path = r"H:\04_前沢SA_上\201909\20190914\カメラ1\2019-09-14_17-00-00.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(fps, length, height, width)

frame_index = 50000

if frame_index >=0 and frame_index < length:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

ret, img = cap.read()

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
    if score >= 0.0 and class_id in [3, 8, 6]:
        #y_min, x_min, y_max, x_max = roi
        #cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        color_mask = np.where(mask[:, :, np.newaxis], color[np.newaxis, np.newaxis, :], color_mask)

cv2.imshow("", img)
cv2.waitKey(0)

cv2.imshow("", color_mask)
cv2.waitKey(0)

img = np.where(color_mask > 0, cv2.addWeighted(img, 0.5, color_mask, 0.5, 0), img)

cv2.imshow("", img)
cv2.waitKey(0)

print(round(end - start, 3))
