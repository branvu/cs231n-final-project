'''
THIS FILE IS FOR ISL DATA PROCESSING
- get 8055 images from folder with uniform distribution
- preprocess with cropping, resizing, and grascaling
- save image to specified folder
- exclude J and Z as they are not static signs
'''

import random
import cv2
from collections import defaultdict
import os
from preprocess_functions import *
import argparse

NUM_LABELS = 24
EXCLUDED_LABELS = {"J", "Z"}
DATA_FOLDER = "data"  # should exist
SAVE_DIR = "isl"  # should NOT exist
MARGIN = 1

# where images are initially located, usually a temp folder
IMG_ORIGIN_FOLDER = "data/isl_original_data"

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
        counter[file[8]] += 1
    dic = [{x: counter[x]} for x in sorted(counter)]
    print(dic)
    print("Total:", sum([counter[x] for x in counter]))


def parse_images(path, show=False, show_crop=False, show_bounding=False):
    images = os.listdir(path)

    counter = defaultdict(int)

    imgs = []
    labels = []
    unused_idxs = set()
    used_idxs = set()

    for i in range(len(images)):
        img_name = images[i]
        label = img_name[8]
        if counter[label] < (TOTAL_WANTED // NUM_LABELS) + 1 and label not in EXCLUDED_LABELS:
            img = cv2.imread(IMG_ORIGIN_FOLDER + "\\" + img_name)
            img = grayscale(
                resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, crop_img(img, MARGIN, show_crop, show_bounding)))

            imgs.append(img)

            if show:
                cv2.imshow("image", img)
                cv2.waitKey(0)

            assert img.shape == (28, 28)

            labels.append(ord(label) - ord('A'))
            counter[label] += 1

            assert i not in used_idxs
            used_idxs.add(i)

        elif label not in EXCLUDED_LABELS:
            unused_idxs.add(i)

    for _ in range(TOTAL_WANTED - len(imgs)):
        idx = random.choice(tuple(unused_idxs))
        assert idx not in used_idxs
        used_idxs.add(idx)

        img_name = images[idx]
        label = img_name[8]
        img = cv2.imread(IMG_ORIGIN_FOLDER + "\\" + img_name)
        img = grayscale(resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE,
                               crop_img(img, 1)))
        imgs.append(img)

        if show:
            cv2.imshow("image", img)
            cv2.waitKey(0)

        assert img.shape == (28, 28)

        labels.append(ord(label) - ord('A'))
        counter[label] += 1
        unused_idxs.remove(idx)

    assert counter['J'] == 0 and counter['Z'] == 0
    return imgs, labels


def main():
    # show statistics and distributions of letters
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
        cv2.imwrite("isl-" + str(labels[i]) + "-" + str(i) + ".jpg", img)

        if i % 100 == 0:
            print("image:", i)

    print(f"Completed {len(images)} images")


if __name__ == "__main__":
    main()
