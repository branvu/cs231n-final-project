'''
THIS FILE IS FOR JSL DATA PROCESSING
- get 8055 images from folder with uniform distribution
- save image to specified folder with name format
- preprocess with cropping, resizing, and grascaling
'''

import cv2
from collections import defaultdict
import os
from preprocess_functions import *
import argparse

NUM_LABELS = 41
DATA_FOLDER = "data"  # should exist
SAVE_DIR = "jsl"  # should NOT exist
MARGIN = 5

IMG_ORIGIN_FOLDER = "data/jsl_original_data"

parser = argparse.ArgumentParser()
parser.add_argument(
    "show", help="Show images (y/N)", nargs='?', const='y')
parser.add_argument(
    "show_crop", help="Show cropping of images (y/N)", nargs='?', const='y')
parser.add_argument(
    "show_bounding", help="Show bounding boxes of images (y/N)", nargs='?', const='y')
args = parser.parse_args()


def statistics(path):
    # get distribution of each label in original folder
    files = os.listdir(path)
    counter = defaultdict(int)

    for file in files:
        counter[file[9:11]] += 1

    dic = [{x: counter[x]} for x in sorted(counter)]
    print(dic)
    print("Total:", sum([counter[x] for x in counter]))


def parse_images(path, show=False, show_crop=False, show_bounding=False):
    images = os.listdir(path)

    imgs = []
    labels = []

    for i in range(len(images)):
        img_name = images[i]
        label = int(img_name[9:11])

        img = cv2.imread(IMG_ORIGIN_FOLDER + "\\" + img_name)

        assert img.shape[0] == img.shape[1]

        img = grayscale(
            resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, crop_img(img, MARGIN, show_crop, show_bounding)))

        if show:
            cv2.imshow("image", img)
            cv2.waitKey(0)

        assert img.shape == (28, 28)

        imgs.append(img)
        labels.append(label)

    return imgs, labels


def main():
    # show statistics and distribution of letters
    statistics(IMG_ORIGIN_FOLDER)

    show = True if args.show is not None and args.show.lower() == 'y' else False
    show_crop = True if args.show_crop is not None and args.show_crop.lower() == 'y' else False
    show_bounding = True if args.show_bounding is not None and args.show_bounding.lower() == 'y' else False

    images, labels = parse_images(
        IMG_ORIGIN_FOLDER, show, show_crop, show_bounding)

    # move and create directories
    assert os.path.isdir(DATA_FOLDER)
    os.chdir(DATA_FOLDER)

    assert not os.path.isdir(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    os.chdir(SAVE_DIR)

    for i in range(len(images)):
        img = images[i]
        # offset of 1 on label bc not 0 indexed
        cv2.imwrite("jsl-" + str(labels[i] - 1) + "-" + str(i) + ".jpg", img)

        if i % 100 == 0:
            print("image:", i)

    print(f"Completed {len(images)} images")


if __name__ == "__main__":
    main()
