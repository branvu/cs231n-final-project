import os
import cv2
from preprocess_functions import *
from math import floor, ceil

DATA_FOLDER = "data"  # should exist
SAVE_DIR = "test_asl_uncropped"  # should NOT exist


def crop_img(image_name, x1, y1, x2, y2):
    img = cv2.imread("../roboflow_asl/" + image_name)
    img = img[y1:y2, x1:x2]
    img = grayscale(resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, img))
    return img


def crop():
    images = []
    with open("../bounding_boxes.txt") as f:
        data = f.readlines()

        for i in range(len(data)):
            coords = data[i].split(",")
            image_name, x1, y1, x2, y2 = coords
            image_name = image_name[image_name.find("valid/") + 6:]

            if image_name[0] == 'J' or image_name[0] == 'Z':
                continue

            label = ord(image_name[0]) - ord('A')

            # img = crop_img(image_name, floor(float(x1)), floor(
            # float(y1)), ceil(float(x2)), ceil(float(y2)))
            img = grayscale(resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE,
                                   cv2.imread("../roboflow_asl/" + image_name)))

            images.append((img, label))
    return images


def main():
    assert os.path.isdir(DATA_FOLDER)
    os.chdir(DATA_FOLDER)

    assert not os.path.isdir(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    os.chdir(SAVE_DIR)

    images = crop()

    for i in range(len(images)):
        img, label = images[i]

        cv2.imwrite(SAVE_DIR + "-" +
                    str(label) + "-" + str(i) + ".jpg", img)

        if i % 100 == 0:
            print("image:", i)

    print(f"Completed {len(images)} images")


if __name__ == "__main__":
    main()


# {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
#     'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
