import torch
import torchvision.transforms as T
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models import *
import os
import cv2
from preprocess_functions import *
import argparse

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

lang = "asl"

NUM_CHARS = 26 + 41 + 26 + 32
NUM_LANGS = 4
lang_char_counts = {0: 24, 1: 41, 2: 24, 3: 32}
lang_id = {'asl': 0, 'jsl': 1, 'isl': 2, 'arsl': 3}
DATA_PATH = f"data/{lang}"


parser = argparse.ArgumentParser()
parser.add_argument(
    "mode", help=f"'lang', 'letter', or 'combo")
args = parser.parse_args()


def preprocess(img, size=28):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Lambda(lambda x: x[:, :, :, None]),
    ])
    img = transform(img)
    return img


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None

    x = model(X)
    correct_vals = x.gather(1, y.view(-1, 1)).squeeze()
    correct_vals = correct_vals.sum()
    correct_vals = correct_vals.backward()
    gradient = X.grad
    gradient = torch.abs(gradient)
    saliency, _ = torch.max(gradient, 3)

    return saliency


def show_saliency_maps(X, y, model, total):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(resize(100, 100, X[i]))
        plt.axis('off')
        if args.mode == "lang":
            title = lang
        else:
            title = y[i] - total
            if lang == "asl":
                title = chr(ord('A') + y[i])

        plt.title(title)
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


def get_data():
    images_list = os.listdir(DATA_PATH)
    images = random.sample(images_list, 5)
    y = []
    X = []
    if args.mode == "combo":
        total = sum([lang_char_counts[i]
                     for i in range(int(lang_id[lang]))])
    else:
        total = 0

    for i in images:
        orig_img = cv2.imread(DATA_PATH + "/" + i)
        img = grayscale(
            resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, orig_img))
        X.append(np.array(img))
        if args.mode == "lang":
            correct_val = lang_id[lang]
        else:
            correct_val = total + int(i.split('-')[1])
        y.append(correct_val)
    X = np.array(X)
    y = np.array(y)
    return X, y, total


def main():
    if args.mode == "combo":
        MODEL_SAVE_PATH = "models\model_combo.pth"
        model = ComboModel(NUM_CHARS)
    elif args.mode == "lang":
        MODEL_SAVE_PATH = "models\model_basic.pth"
        model = LanguageModel(NUM_LANGS)
    else:
        MODEL_SAVE_PATH = f"models\model_basic_{lang}_letter.pth"
        if lang == "asl":
            model = ASLModel()
        elif lang == "jsl":
            model = JSLModel()
        elif lang == "isl":
            model = ISLModel()
        elif lang == "arsl":
            model = ARSLModel()
        else:
            raise Exception("Invalid language")

    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint)

    X, y, total = get_data()

    show_saliency_maps(X, y, model, total)


if __name__ == "__main__":
    main()
