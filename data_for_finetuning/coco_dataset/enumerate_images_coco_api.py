import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from skimage import io as io
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_file_path", type=str, default=r"D:\COCO_dataset\annotations_trainval2017\annotations\instances_train2017.json", help="Path to label json file")
    parser.add_argument("--considered_classes", type=str, default="bicycle,motorcycle,car,bus,truck")
    parser.add_argument("--save_dir", type=str, default=r"D:\COCO_dataset\COCO_2017", help="Images directory")
    parser.add_argument("--output_label_path", type=str, default=r"truncated_val2017.json", help="Output label json")

    args = parser.parse_args()

    return args


def showAnns(args, img, anns):
    image = cv2.imread(os.path.join(args.save_dir, img["file_name"]))
    for ann in anns:
        assert img["id"] == ann["image_id"]
        if type(ann["segmentation"]) == list:
            bbox = ann["bbox"]
            segmentation = ann["segmentation"]
            for segment in segmentation:
                segment = np.array(segment, dtype=np.uint16).reshape(-1, 2).tolist()
                for j, point in enumerate(segment):
                    x1, y1 = point
                    if j < len(segment) - 1:
                        x2, y2 = segment[j + 1]
                    else:
                        x2, y2 = segment[0]
                    cv2.line(image, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=1)
                x, y, w, h = list(map(lambda x: int(x), bbox))
                cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 255, 0), thickness=0)

        else:
            if type(ann["segmentation"]["counts"]) == list:
                rle = maskUtils.frPyObjects([ann["segmentation"]], img["height"], img["width"])
            else:
                rle = [ann["segmentation"]]

            m = maskUtils.decode(rle)

            img_mask = np.ones([m.shape[0], m.shape[1], 3])

            if ann["iscrowd"] == 1:
                color_mask = np.array([2.0, 166.0, 101.0]) / 255

            if ann['iscrowd'] == 0:
                color_mask = np.random.random((1, 3)).tolist()[0]

            for i in range(3):
                img_mask[:, :, i] = color_mask[i]

            cv2.imshow("img_mask", np.dstack((img_mask, m * 0.5)))
            cv2.waitKey(0)
    cv2.imshow("img", image)
    cv2.waitKey(0)



if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    coco = COCO(annotation_file=args.label_file_path)

    categories = coco.loadCats(ids=coco.getCatIds())

    print("categories: {}".format(categories))

    names = [category["name"] for category in categories]

    print("COCO categories: \n{}\n".format(" ".join(names)))

    considered_classes = args.considered_classes.split(",")

    catIds = coco.getCatIds(catNms=["motorcycle"])

    print("catIds: {}".format(catIds))

    imgIds = coco.getImgIds(catIds=catIds)

    print("imgIds: {}".format(imgIds))

    coco.download(tarDir=args.save_dir, imgIds=imgIds)

    imgs = coco.loadImgs(ids=imgIds)

    print("imgs: {}".format(imgs))

    for img in imgs:

        annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds, iscrowd=None)

        anns = coco.loadAnns(ids=annIds)

        showAnns(args, img, anns)
