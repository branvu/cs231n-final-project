'''
THIS FILE IS FOR TESTING TRAINED ASL MODEL ON TEST IMAGES FROM ANOTHER DATASET NOT USED IN TRAINING
'''

import random
import torch
from models import *
import os
from preprocess_functions import *
import argparse


# parse argument of language to train
MODEL_SAVE_PATH = "models/model_basic_asl_letter.pth"
DATA_PATH = "data/test_data/asl_alphabet_train"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_CHARS = 24 + 41 + 24 + 32

LABELS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 100000, 10: 9, 11: 10, 12: 11, 13: 12,
          14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 100000}

parser = argparse.ArgumentParser()
parser.add_argument(
    "verbose", help=f"Show images")
args = parser.parse_args()


def get_data():
    images_list = os.listdir(DATA_PATH)
    images = random.sample(images_list, 200)
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

            orig_label = LABELS[ord(image_name[0]) - ord('A')]
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

    model = ASLModel()
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint)
    model.to(device)
    if args.verbose == "true":
        check_accuracy(images, model, True)
    else:
        check_accuracy(images, model, False)


if __name__ == "__main__":
    main()
