'''
THIS FILE IS FOR JSL DATA PROCESSING
- get 8055 images from folder with uniform distribution
- save image to specified folder with name format
'''

import cv2
from collections import defaultdict
import os

FINAL_IMAGE_SIZE = 28
NUM_LABELS = 41
TOTAL_WANTED = 8055
DATA_FOLDER = "data"  # should exist
SAVE_DIR = "jsl"  # should NOT exist

IMG_ORIGIN_FOLDER = "jsl"


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
        counter[file[9:11]] += 1

    print(counter)
    print("Total:", sum([counter[x] for x in counter]))


def grayscale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_gray


def parse_images(path):
    images = os.listdir(IMG_ORIGIN_FOLDER)

    imgs = []
    labels = []

    for i in range(len(images)):
        img_name = images[i]
        label = int(img_name[9:11])

        img = cv2.imread(IMG_ORIGIN_FOLDER + "\\" + img_name)

        assert img.shape[0] == img.shape[1]

        img = grayscale(resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, img))

        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        assert img.shape == (28, 28)

        imgs.append(img)
        labels.append(label)

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
        cv2.imwrite("jsl-" + str(labels[i]) + "-" + str(i) + ".jpg", img)

        if i % 100 == 0:
            print("image:", i)

    print(f"Completed {len(images)} images")


if __name__ == "__main__":
    main()
