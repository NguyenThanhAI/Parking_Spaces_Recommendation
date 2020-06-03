import os
import argparse
import shutil
from tqdm import tqdm
import numpy as np

np.random.seed(1000)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True, help="Directory contains images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory contains unpacked directories")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    images_list = list()

    for dirs, _, images in os.walk(args.input_dir):
        for image in images:
            if image.endswith(".jpg"):
                images_list.append(image)

    np.random.shuffle(images_list)

    num_dirs = int(np.ceil(len(images_list) / args.chunk_size))

    for i in tqdm(range(num_dirs)):
        unpacked_images_list = images_list[i * args.chunk_size: (i + 1) * args.chunk_size]
        print(len(unpacked_images_list))
        if not os.path.exists(os.path.join(args.output_dir, "packed_" + str(i + 1))):
            os.makedirs(os.path.join(args.output_dir, "packed_" + str(i + 1)))

        for image in unpacked_images_list:
            shutil.copy(os.path.join(args.input_dir, image), os.path.join(args.output_dir, "packed_" + str(i + 1), image))
