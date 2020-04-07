import sys
import os
import argparse
import time
import numpy as np
import imgaug
import cv2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT_DIR = os.path.abspath("../")

print("{}".format(ROOT_DIR))

sys.path.append(ROOT_DIR)

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "test_object_detection_models", "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "finetuning", "logs")

print("{}, {}".format(COCO_MODEL_PATH, DEFAULT_LOGS_DIR))


class CarsAndVehiclesConfig(Config):

    NAME = "cars_and_vehicles"

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1 # Background + car

    STEPS_PER_EPOCH = 2000

    DETECTION_MIN_CONFIDENCE = 0.6


class CarsAndVehiclesDataset(utils.Dataset):
    def load_carsandvehicles(self,
                             dataset_dir,
                             subset,
                             json_label_file_path,
                             class_ids=None,
                             class_map=None,
                             return_coco=False,
                             ):

        coco = COCO(os.path.join(dataset_dir, json_label_file_path))

        print("coco: {}".format(coco))

        image_dir = os.path.join(dataset_dir, "finetuning_dataset")

        if not class_ids:
            class_ids = sorted(coco.getCatIds())

        print("class_ids: {}".format(class_ids))

        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))

            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        print("image_ids: {}".format(image_ids))

        # Add classes
        for i in class_ids:
            self.add_class(source="self_annotation", class_id=i, class_name=coco.loadCats(i)[0]["name"])
            print("class_id={}, class_name={}".format(i, coco.loadCats(i)[0]["name"]))

        for i in image_ids:
            self.add_image("self_annotation",
                           image_id=i,
                           path=os.path.join(image_dir, coco.imgs[i]["file_name"]),
                           width=coco.imgs[i]["width"],
                           height=coco.imgs[i]["height"],
                           annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))

        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances]
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "self_annotation":
            super(CarsAndVehiclesDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id("self_annotation.{}".format(annotation["category_id"]))

            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])

                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID
                if annotation["iscrowd"]:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones(image_info["height"], image_info["width"], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CarsAndVehiclesDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return name of image in finetuning dataset"""
        info = self.image_info[image_id]
        if info["source"] == "self_annotation":
            return os.path.basename(info["path"])
        else:
            super(CarsAndVehiclesDataset, self).image_reference(image_id)

    # The following two functions are form pycocotools with a few changes

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann["segmentation"]
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm["counts"], list):
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            rle = ann["segmentation"]
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)

        return m

dataset = CarsAndVehiclesDataset()
dataset.load_carsandvehicles(dataset_dir=r"F:\Parking_Spaces_Recommendation_Data",
                             subset="train",
                             json_label_file_path="Label_car.json")
dataset.prepare()
masks, class_ids = dataset.load_mask(image_id=0)

print("Masks: {}, Class_ids: {}".format(masks.shape, class_ids.shape))

masks = np.max(masks.astype(np.uint8), axis=-1)

masks = masks * 255

cv2.imshow("", masks)
cv2.waitKey(0)

ref = dataset.image_reference(image_id=0)
print("Image reference: {}".format(ref))