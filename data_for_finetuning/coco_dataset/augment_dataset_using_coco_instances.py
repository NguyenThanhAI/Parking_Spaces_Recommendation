import os
import argparse
from itertools import groupby, chain
from operator import itemgetter
import json
import numpy as np
import cv2
from skimage.transform import rescale
from skimage.draw import polygon


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_dataset_dir", type=str, default=r"D:\finetuning_dataset", help="Original dataset to finetuning")
    parser.add_argument("--original_label_file", type=str, default=r"C:\Users\Thanh\Downloads\Annotations\unzip\Label.json", help="Original annotated json file")
    parser.add_argument("--augmenting_dataset_dir", type=str, default=r"D:\COCO_dataset\COCO_2017\train", help="COCO dataset to augment finetuning dataset")
    parser.add_argument("--augmenting_label_file", type=str, default=r"D:\COCO_dataset\annotations_trainval2017\annotations\truncated_train2017.json", help="Annotation file of COCO with desired label need to be augmented")
    parser.add_argument("--augmented_dataset_dir", type=str, default=r"D:\augmented_finetuning_dataset\finetuning_dataset", help="Output of augmented images directory")
    parser.add_argument("--augmented_label_file", type=str, default=r"Augmented_Label.json", help="NAME of output augmented annotated file")
    parser.add_argument("--min_area_instance_to_choose", type=int, default=100000, help="Min area of instance to choose to agument original dataset")
    parser.add_argument("--min_vertical_image_position", type=int, default=0, help="Min position against y axis to put instance on image")
    parser.add_argument("--max_vertical_image_position", type=int, default=550, help="Min position against y axis to put instance on image")
    parser.add_argument("--min_horizontal_image_position", type=int, default=0, help="Min position against x axis to put instance on image")
    parser.add_argument("--max_horizontal_image_position", type=int, default=1000, help="Min position against x axis to put instance on image")
    parser.add_argument("--min_vertical_dim", type=int, default=20, help="Min dimensional along vertical dimension")
    parser.add_argument("--max_vertical_dim", type=int, default=100, help="Max dimensional along vertical dimension")
    parser.add_argument("--max_num_augmented_instances_per_image", type=int, default=10, help="Max number of augmented instances per image")
    parser.add_argument("--min_width", type=int, default=400, help="Min width of instance to be considered")
    parser.add_argument("--min_height", type=int, default=400, help="Min height of instance to be considered")

    args = parser.parse_args()

    return args


def read_label_file(label_file_path):
    with open(label_file_path, "r") as f:
        json_label = json.load(f)

    return json_label


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.augmented_dataset_dir):
        os.makedirs(args.augmented_dataset_dir, exist_ok=True)

    original_label_file = read_label_file(args.original_label_file)
    augmenting_label_file = read_label_file(args.augmenting_label_file)

    original_images = original_label_file["images"]
    original_annotations = original_label_file["annotations"]
    categories = original_label_file["categories"]
    info = original_label_file["info"]
    licenses = original_label_file["licenses"]

    assert isinstance(original_images, list) and isinstance(original_annotations, list)

    augmenting_images = augmenting_label_file["images"]
    augmenting_annotations = augmenting_label_file["annotations"]

    assert isinstance(augmenting_images, list) and isinstance(augmenting_annotations, list)

    #augmenting_annotations = list(filter(lambda x: x["area"] >  args.min_area_instance_to_choose and isinstance(x["segmentation"], list) and x["iscrowd"] == 0 and isinstance(x["bbox"][2], float) and isinstance(x["bbox"][3], float) and x["bbox"][2] > args.min_width and x["bbox"][3] > args.min_height, augmenting_annotations))


    augmented_images = []
    augmented_annotations = []

    anno_id = 0
    for image_id, items in groupby(original_images, key=lambda x: x["id"]):
        print("image_id {}".format(image_id))
        for item in items:
            file_name = item["file_name"]
            width = item["width"]
            height = item["height"]
            file_path = os.path.join(args.original_dataset_dir, file_name)
            augmented_images.append(item)

        img = cv2.imread(file_path)

        cars_or_vehicles = list(filter(lambda x: x["image_id"] == image_id, original_annotations))

        for car_or_vehicle in cars_or_vehicles:
            car_or_vehicle["id"] = anno_id
            augmented_annotations.append(car_or_vehicle)
            anno_id += 1

        num_augmented_instances = np.random.randint(low=2, high=args.max_num_augmented_instances_per_image)
        print("num_augmented_instances {}".format(num_augmented_instances))
        added_instances = np.random.choice(augmenting_annotations, size=num_augmented_instances, replace=True)
        print("num chosen instances: {}, {}".format(len(added_instances), len(augmenting_annotations)))
        augmented_mask = np.zeros_like(img, dtype=np.uint8)

        for i, instance in enumerate(added_instances):
            segmentation = instance["segmentation"]
            area = instance["area"]
            bbox = instance["bbox"]
            #if len(segmentation) != 1 or area <= args.min_area_instance_to_choose or bbox[2] <= args.min_width and bbox[3] <= args.min_height:
            #    continue
            if len(segmentation) != 1:
                continue

            corresponding_image = list(filter(lambda x: x["id"] == instance["image_id"], augmenting_images))

            assert len(corresponding_image) == 1

            corresponding_image = corresponding_image[0]["file_name"]
            corresponding_image = cv2.imread(os.path.join(args.augmenting_dataset_dir, corresponding_image))

            segmentation = np.array(segmentation, dtype=np.uint16).reshape(-1, 2)
            x_min, y_min = np.min(segmentation, axis=0)
            x_max, y_max = np.max(segmentation, axis=0)
            segmentation = segmentation - np.array([x_min, y_min])[np.newaxis, :]

            cropped_instance = corresponding_image[y_min: y_max + 1, x_min: x_max + 1].copy()
            #print("shape of instance {}, bbox {}".format(cropped_instance.shape, instance["bbox"]))
            if np.prod(cropped_instance.shape) == 0:
                continue
            y_position = np.random.randint(low=args.min_vertical_image_position, high=args.max_vertical_image_position)
            x_position = np.random.randint(low=args.min_horizontal_image_position, high=args.max_horizontal_image_position)
            #print(y_position, x_position)
            target_size = (y_position - args.min_vertical_image_position) * (args.max_vertical_dim - args.min_vertical_dim) / (args.max_vertical_image_position - args.min_vertical_image_position) + args.min_vertical_dim

            scale_factor = target_size / cropped_instance.shape[0]
            #print("scale_factor {}".format(scale_factor))
            segmentation = (segmentation * scale_factor).astype(int) #- np.array([1])[np.newaxis, np.newaxis]
            area = area * (scale_factor ** 2)

            cc, rr = segmentation.T

            rr, cc = polygon(rr, cc)

            rr = np.where(rr >= cropped_instance.shape[0], cropped_instance.shape[0] - 1, rr)
            cc = np.where(cc >= cropped_instance.shape[1], cropped_instance.shape[1] - 1, cc)

            cropped_instance = rescale(cropped_instance, scale=scale_factor)

            mask = np.zeros_like(cropped_instance, dtype=np.bool)

            #try:
            mask[rr, cc] = True
            #except Exception as e:
            #    print(e)
            #    continue

            cropped_instance = (np.where(mask, cropped_instance, np.zeros_like(cropped_instance)) * 255.).astype(np.uint8)

            augmented_mask[y_position:(y_position + cropped_instance.shape[0]), x_position:(x_position + cropped_instance.shape[1])] = cropped_instance

            segmentation = segmentation + np.array([x_position, y_position], dtype=np.float)[np.newaxis, :]
            cc, rr = segmentation.T
            cc = cc.tolist()
            rr = rr.tolist()
            segmentation = [list(chain(*list(zip(cc, rr))))]

            bbox = [x_position, y_position, cropped_instance.shape[1], cropped_instance.shape[0]]

            instance["segmentation"] = segmentation
            instance["bbox"] = bbox
            instance["area"] = area
            instance["image_id"] = image_id
            instance["id"] = anno_id

            augmented_annotations.append(instance)

            anno_id += 1
        img = np.where(augmented_mask, augmented_mask, img)
        cv2.imwrite(os.path.join(args.augmented_dataset_dir, file_name), img)

        cv2.imshow("Anh", img)
        cv2.imshow("Mask", augmented_mask)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    augmented_json = {}

    augmented_json["images"] = augmented_images
    augmented_json["annotations"] = augmented_annotations
    augmented_json["licenses"] = licenses
    augmented_json["categories"] = categories
    augmented_json["info"] = info

    with open(os.path.join(os.path.dirname(args.original_label_file), args.augmented_label_file), "w") as f:
        json.dump(augmented_json, f)
