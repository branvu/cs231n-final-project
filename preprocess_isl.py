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


FINAL_IMAGE_SIZE = 28
NUM_LABELS = 24
EXCLUDED_LABELS = {"J", "Z"}
TOTAL_WANTED = 8055
DATA_FOLDER = "data"  # should exist
SAVE_DIR = "isl"  # should NOT exist

# where images are initially located, usually a temp folder
IMG_ORIGIN_FOLDER = "Person1"


def crop_img(path):
    # read image
    img = cv2.imread(path)

    # create binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # get contours (bounding boxes)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # should only only be one box
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if len(cnts) != 1:
        cnts = cnts[-1:]

        # display bounding boxes
        # for c in cnts:
        #     x, y, w, h = cv2.boundingRect(c)
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     cv2.imshow("Resized image", img)
        #     cv2.waitKey(0)

    assert len(cnts) == 1

    # extract coords from bounding box
    x, y, w, h = cv2.boundingRect(cnts[0])

    # get square image dimensions
    edge_length = max(w, h)
    x = x + w//2 - edge_length//2
    y = y + h//2 - edge_length//2

    # crop image
    img = img[y:y+edge_length, x:x+edge_length]

    return img


def resize(w, h, img):
    dim = (w, h)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def statistics(path):
    # get distribution of each label in original folder

    files = os.listdir(path)
    counter = defaultdict(int)

    for file in files:
        counter[file[8]] += 1
    print(counter)
    print("Total:", sum([counter[x] for x in counter]))


def grayscale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def parse_images(path):
    images = os.listdir(IMG_ORIGIN_FOLDER)

    counter = defaultdict(int)

    imgs = []
    labels = []
    unused_idxs = set()
    used_idxs = set()

    for i in range(len(images)):
        img_name = images[i]
        label = img_name[8]
        if counter[label] < (TOTAL_WANTED // NUM_LABELS) + 1 and label not in EXCLUDED_LABELS:
            img = grayscale(
                resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, crop_img(IMG_ORIGIN_FOLDER + "\\" + img_name)))

            imgs.append(img)
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
        img = grayscale(resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE,
                               crop_img(IMG_ORIGIN_FOLDER + "\\" + img_name)))
        imgs.append(img)
        labels.append(ord(label) - ord('A'))
        counter[label] += 1
        unused_idxs.remove(idx)

    assert counter['J'] == 0 and counter['Z'] == 0
    return imgs, labels


def main():
    statistics(IMG_ORIGIN_FOLDER)
    images, labels = parse_images(IMG_ORIGIN_FOLDER)

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
