'''
THIS FILE IS FOR TESTING TRAINED ASL MODEL ON TEST IMAGES FROM ANOTHER DATASET NOT USED IN TRAINING
'''

import random
import torch
from models import *
import os
from preprocess_functions import *


# parse argument of language to train
MODEL_SAVE_PATH = "models/model_basic_asl_letter.pth"
DATA_PATH = "data/test_data/asl_alphabet_train"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
            # move to device
            orig_img = cv2.imread(DATA_PATH + "/" + image_name)
            orig_img = grayscale(
                resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, orig_img))
            img = orig_img[None, :, :, None]

            input = torch.tensor(img)
            input = input.to(device)

            orig_label = ord(image_name[0]) - ord('A')
            label = torch.tensor([orig_label])

            letter = label.to(device)

            out = model(input)
            _, predicted = torch.max(out.data, 1)

            if verbose:
                print(f"Predicted: {predicted}, Label: {orig_label}")
                cv2.imshow("img", orig_img)
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
    check_accuracy(images, model, True)


if __name__ == "__main__":
    main()
