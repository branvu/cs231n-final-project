'''
THIS FILE IS FOR TESTING TRAINED ASL MODEL ON TEST IMAGES FROM ANOTHER DATASET NOT USED IN TRAINING
'''

import random
import torch
from models import *
import os
from preprocess_functions import *
import argparse
import time

# parse argument of language to train
MODEL_SAVE_PATH = "models/model_combo.pth"
DATA_PATH = "data/test_asl"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_CHARS = 26 + 41 + 26 + 32

LABELS = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
          'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

parser = argparse.ArgumentParser()
parser.add_argument(
    "verbose", help=f"Show images")
args = parser.parse_args()


def get_data():
    images = os.listdir(DATA_PATH)
    # images = random.sample(images, 200)
    return images


def check_accuracy(images, model, verbose=False):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for image_name in images:
            # print(image_name)
            # move to device
            orig_img = cv2.imread(DATA_PATH + "/" + image_name)
            orig_img = grayscale(
                resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, orig_img))
            img = orig_img[None, :, :, None]

            input = torch.tensor(img)
            input = input.to(device)

            orig_label = LABELS[chr(ord('A') + int(image_name.split('-')[1]))]
            # print(orig_label)
            label = torch.tensor([orig_label])

            letter = label.to(device)
            # print(orig_label, letter)
            out = model(input)
            _, predicted = torch.max(out.data, 1)

            if verbose:
                print(f"Predicted: {predicted}, Label: {orig_label}")
                cv2.imshow("img", resize(100, 100, orig_img))
                cv2.waitKey(0)

            correct += (predicted == letter).sum().item()
            total += predicted.size(0)
        acc = float(correct) / total
        print('Got %d / %d correct (%.2f)' % (correct, total, 100 * acc))


def main():
    images = get_data()

    model = ComboModel(NUM_CHARS)
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint)
    model.to(device)
    if args.verbose == "true":
        check_accuracy(images, model, True)
    else:
        start = time.time()
        check_accuracy(images, model, False)
        print((time.time() - start)/len(images))


if __name__ == "__main__":
    main()
