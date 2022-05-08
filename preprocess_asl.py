'''
THIS FILE IS FOR ASL DATASET PROCESSING
- read images from csv format
- get 8055 images with uniform distribution over letters
- save as .jpg file in folder specified
- exclude J and Z as they are not static signs
- original data is 28x28 and grayscale
'''

import random
import cv2
import numpy as np
import csv
from collections import defaultdict
import os
from preprocess_functions import *
import argparse
import pandas as pd
from collections import Counter

NUM_LABELS = 24  # doesn't contain J and Z datapoints in original csv file
DATA_FOLDER = "data"  # should exist
SAVE_DIR = "asl"  # should NOT exist
IMG_ORIGIN_FOLDER = "data/sign_mnist_train.csv"

parser = argparse.ArgumentParser()
parser.add_argument(
    "show", help="Show images (y/N)", nargs='?', const='y')
args = parser.parse_args()


def statistics(filename):
    df = pd.read_csv(filename)
    labels = df['label']
    counter = Counter(labels)
    dic = [{x: counter[x]} for x in sorted(counter)]
    print(dic)


def parse_data_from_input(filename, show=False):
    counts = defaultdict(int)

    with open(filename) as file:
        reader = csv.reader(file, delimiter=',')
        imgs = []
        labels = []
        next(reader, None)
        for row in reader:
            label = row[0]
            if counts[label] < (TOTAL_WANTED // NUM_LABELS) + 1:
                data = row[1:]
                img = np.asarray(data, dtype=float).reshape((28, 28))
                img = img.astype(np.uint8)

                # image already grayscale and centered
                imgs.append(img)

                if show:
                    cv2.imshow("image", img)
                    cv2.waitKey(0)

                assert img.shape == (28, 28)

                labels.append(label)
                counts[label] += 1

    for _ in range(len(imgs) - TOTAL_WANTED):
        idx = random.randint(0, len(imgs) - 1)
        imgs = imgs[:idx] + imgs[idx + 1:]
        labels = labels[:idx] + labels[idx + 1:]

    images = np.array(imgs).astype(np.uint8)
    labels = np.array(labels).astype(np.uint8)
    return images, labels


def main():
    statistics(IMG_ORIGIN_FOLDER)

    show = True if args.show is not None and args.show.lower() == 'y' else False
    images, labels = parse_data_from_input(IMG_ORIGIN_FOLDER, show)

    # move and create directories
    assert os.path.isdir(DATA_FOLDER)
    os.chdir(DATA_FOLDER)

    assert not os.path.isdir(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    os.chdir(SAVE_DIR)

    # save images
    for i in range(len(images)):
        img = images[i]
        cv2.imwrite("asl-" + str(labels[i]) + "-" + str(i) + ".jpg", img)

        if i % 100 == 0:
            print("image:", i)

    print(f"Completed {len(images)} images")


if __name__ == "__main__":
    main()
