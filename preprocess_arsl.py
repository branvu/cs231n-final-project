'''
THIS FILE IS FOR ARSL DATA PROCESSING
- get 8055 images from folder with uniform distribution
- preprocess with cropping, resizing, and grascaling
- save image to specified folder
'''

from cProfile import label
import random
import cv2
from collections import defaultdict
import os
from preprocess_functions import *
import argparse
import pandas as pd

NUM_LABELS = 32
DATA_FOLDER = "data"  # should exist
SAVE_DIR = "arsl"  # should NOT exist
MARGIN = 3
LABEL_MAP, LABELS = None, None

# where images are initially located, usually a temp folder
IMG_ORIGIN_FOLDER = "data/arsl_original_data"

parser = argparse.ArgumentParser()
parser.add_argument(
    "show", help="Show images (y/N)", nargs='?', const='y')
parser.add_argument(
    "show_crop", help="Show cropping of images (y/N)", nargs='?', const='y')
parser.add_argument(
    "show_bounding", help="Show bounding boxes of images (y/N)", nargs='?', const='y')
args = parser.parse_args()


def statistics(path):
    global LABELS, LABEL_MAP

    # get distribution of each label in original folder
    LABELS = os.listdir(path)

    LABEL_MAP = {LABELS[x]: x for x in range(len(LABELS))}
    counter = defaultdict(int)

    for folder in LABELS:
        num = len(os.listdir(path + "/" + folder))
        counter[LABEL_MAP[folder]] = num
    dic = [{x: counter[x]} for x in sorted(counter)]

    print(dic)
    print("Total:", sum([counter[x] for x in counter]))

    # ensure labels are correct
    df = pd.read_csv("ArSL_Data_Labels.csv")
    classes = set(df['Class'])
    assert len(classes) == NUM_LABELS

    LABELS = set(LABELS)
    assert len(LABELS) == NUM_LABELS

    assert len(classes.difference(LABELS)) == 0


def parse_images(path, show=False, show_crop=False, show_bounding=False):
    folders = os.listdir(path)

    counter = defaultdict(int)

    imgs = []
    labels = []

    # go through the folders for each letter
    for folder in folders:
        images = os.listdir(path + "/" + folder)

        # get images for each label
        for i in range(TOTAL_WANTED // NUM_LABELS + 1):
            img_name = images[i]
            label = LABEL_MAP[folder]

            img = cv2.imread(path + "/" + folder + "/" + img_name)
            img = grayscale(resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, crop_img(
                img, MARGIN, show_crop, show_bounding)))

            imgs.append(img)

            if show:
                cv2.imshow("image", img)
                cv2.waitKey(0)

            assert img.shape == (28, 28)

            labels.append(label)
            counter[label] += 1

    for _ in range(len(imgs) - TOTAL_WANTED):
        idx = random.randint(0, len(imgs) - 1)
        imgs = imgs[:idx] + imgs[idx + 1:]
        labels = labels[:idx] + labels[idx + 1:]
        counter[label] -= 1

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
        cv2.imwrite(SAVE_DIR + "-" +
                    str(labels[i]) + "-" + str(i) + ".jpg", img)

        if i % 100 == 0:
            print("image:", i)

    print(f"Completed {len(images)} images")


if __name__ == "__main__":
    main()
