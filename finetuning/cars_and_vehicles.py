import os
import sys
import argparse
import time
import numpy as np
import imgaug

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "test_object_detection_models", "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "finetuning", "logs")


class CarsAndVehiclesConfig(Config):

    NAME = "cars_and_vehicles"

    GPU_COUNT = 2

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 4 # Background + car

    STEPS_PER_EPOCH = 500

    DETECTION_MIN_CONFIDENCE = 0.6


############################################################
#  Dataset
############################################################


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

        image_dir = os.path.join(dataset_dir, "finetuning_dataset")

        if not class_ids:
            class_ids = sorted(coco.getCatIds())

        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))

            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class(source="self_annotation", class_id=i, class_name=coco.loadCats(i)[0]["name"])

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

############################################################
#  COCO Evaluation
############################################################


def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
        """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []

    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "self_annotation"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)

    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
        dataset: A Dataset object with validation data
        eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
        limit: if not 0, it's the number of images to use for evaluation
        """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get correspoding COCO image IDs
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []

    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        t = time.time()
        r = model.detect([image], verbose=0)[0]

        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool

        image_results = build_coco_results(dataset,
                                           coco_image_ids[i:i + 1],
                                           r["rois"],
                                           r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))

        results.extend(image_results)

    # Load results. This modifies results with additional attributes.

    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.ImgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


def train(model, args):
    """Train the model"""
    dataset_train = CarsAndVehiclesDataset()
    dataset_train.load_carsandvehicles(dataset_dir=args.dataset,
                                       subset="train",
                                       json_label_file_path=args.train_json_label_file_path)
    dataset_train.prepare()

    dataset_val = CarsAndVehiclesDataset()
    dataset_val.load_carsandvehicles(dataset_dir=args.dataset,
                                     subset="val",
                                     json_label_file_path=args.test_json_label_file_path)
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # Training - Stage 1
    print("Training the last layer")
    model.train(train_dataset=dataset_train,
                val_dataset=dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers="heads",
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 5 and up
    print("Finetune Resnet stage 5 and up")
    model.train(train_dataset=dataset_train,
                val_dataset=dataset_val,
                learning_rate=config.LEARNING_RATE / 2,
                epochs=20,
                layers="5+",
                augmentation=augmentation)


    # Training - Stage 3
    # Finetune layers from ResNet stage 4 and up
    print("Finetune Resnet stage 4 and up")
    model.train(train_dataset=dataset_train,
                val_dataset=dataset_val,
                learning_rate=config.LEARNING_RATE / 4,
                epochs=30,
                layers="4+",
                augmentation=augmentation)

    # Training - Stage 4
    # Finetune layers from ResNet stage 3 and up
    print("Finetune Resnet stage 3 and up")
    model.train(train_dataset=dataset_train,
                val_dataset=dataset_val,
                learning_rate=config.LEARNING_RATE / 4,
                epochs=40,
                layers="3+",
                augmentation=augmentation)

    # Training - Stage 5
    # Finetune all layers
    print("Finetune all layers")
    model.train(train_dataset=dataset_train,
                val_dataset=dataset_val,
                learning_rate=config.LEARNING_RATE / 8,
                epochs=50,
                layers="all",
                augmentation=augmentation)


def evaluate(model, args):
    dataset_val = CarsAndVehiclesDataset()
    coco = dataset_val.load_carsandvehicles(dataset_dir=args.dataset,
                                            subset="val",
                                            json_label_file_path=args.test_json_label_file_path,
                                            return_coco=True)
    dataset_val.prepare()
    print("Running COCO evaluation on {} images".format(args.limit))
    evaluate_coco(model=model,
                  dataset=dataset_val,
                  coco=coco,
                  eval_type="bbox",
                  limit=int(args.limit),)


############################################################
#  Training
############################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train Mask R-CNN on custom dataset.")

    parser.add_argument("command",
                        metavar="<command>",
                        help="train or evaluate")
    parser.add_argument("--dataset",
                        required=True,
                        metavar="/path/to/dataset/",
                        help="Directory of custom dataset")
    parser.add_argument("--model",
                        required=True,
                        metavar="/path/to/mask_rcnn_coco.h5",
                        help="Pretrained weights name")
    parser.add_argument("--train_json_label_file_path",
                        required=False,
                        default="Label_car.json",
                        metavar="/path/to/label/file/",
                        help="Path to json label file")
    parser.add_argument("--test_json_label_file_path",
                        required=False,
                        default="Label_car.json",
                        metavar="/path/to/label/file/",
                        help="Path to json label file")
    parser.add_argument("--logs",
                        required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help="Logs and checkpoints directory")
    parser.add_argument("--limit",
                        required=False,
                        default=1000,
                        metavar="<image count>",
                        help="Maximum number of images to use for evaluation")
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CarsAndVehiclesConfig()
    else:
        class InferenceConfig(CarsAndVehiclesConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training",
                                  config=config,
                                  model_dir=args.logs)

    else:
        model = modellib.MaskRCNN(mode="inference",
                                  config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
        if not os.path.exists(model_path):
            utils.download_trained_weights(model_path)
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights", model_path)
    if args.model.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(model_path,
                           by_name=True,
                           exclude=["mrcnn_class_logits",
                                    "mrcnn_bbox_fc",
                                    "mrcnn_bbox",
                                    "mrcnn_mask"])
    else:
        model.load_weights(model_path,
                           by_name=True)

    # Train or evaluate

    if args.command == "train":
        # Training dataset. Use the training set
        train(model, args)
    elif args.command == "evaluate":
        # Validation dataset
        evaluate(model, args)
    else:
        print("'{}' is not recognized. Use 'train' or 'evaluate'".format(args.command))
