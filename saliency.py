import torch
import torchvision
import torchvision.transforms as T
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from PIL import Image
from models import *
import os
import cv2
from preprocess_functions import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

NUM_CHARS = 24 + 41 + 24 + 32
MODEL_SAVE_PATH = "models/model_combo.pth"
DATA_PATH = "data/asl"
model = ComboModel(NUM_CHARS)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(MODEL_SAVE_PATH)
model.load_state_dict(checkpoint)
# model.to(device)


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
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # forward pass
    # X = X.to(device)
    # scores = model(X)
    # print(scores)
    # # print(scores.shape) # torch.Size([5, 1000]) since 5 images, 1000 classes
    # scores = (scores.gather(1, y.view(-1, 1)).squeeze())
    # print(scores)
    # # print(scores.shape) # torch.Size([5])

    # # print(scores) #tensor([24.1313, 25.1475, 38.8825, 25.4514, 30.2723], grad_fn=<SqueezeBackward0>)

    # # backward pass
    # # https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
    # # print(scores.shape[0]) # 5
    # # print(torch.FloatTensor([1.0]*scores.shape[0])) # tensor([1., 1., 1., 1., 1.])
    # scores.backward(torch.FloatTensor([1.0]*scores.shape[0]))

    # print(X.grad.data.abs().shape)
    # # saliency
    # saliency, _ = torch.max(X.grad.data.abs(), dim=3)

    x = model(X)
    print(x)
    correct_vals = x.gather(1, y.view(-1, 1)).squeeze()
    print(correct_vals)
    correct_vals = correct_vals.sum()
    correct_vals = correct_vals.backward()
    gradient = X.grad
    gradient = torch.abs(gradient)
    saliency, _ = torch.max(gradient, 3)

    print(saliency.shape)

    # torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)

    # Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor
    # in the given dimension dim. And indices is the index location of each maximum value found (argmax).
    # If keepdim is True, the output tensors are of the same size as input except in the dimension dim
    # where they are of size 1. Otherwise, dim is squeezed (see torch.squeeze()),
    # resulting in the output tensors having 1 fewer dimension than input.

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency


def show_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)

    # X_tensor = X_tensor.to(device)
    # y_tensor = y_tensor.to(device)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
    print(N)
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(resize(100, 100, X[i]))
        plt.axis('off')
        plt.title(y[i])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        # print(saliency[i])
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
        # break
    plt.show()


def get_data():
    images_list = os.listdir(DATA_PATH)
    images = random.sample(images_list, 5)
    # images = images_list[6000:6000+5]
    y = []
    X = []
    for i in images:
        orig_img = cv2.imread(DATA_PATH + "/" + i)
        img = grayscale(
            resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, orig_img))
        # img = img[None, :, :, None]
        # print(img)
        X.append(np.array(img))
        y.append(int(i[4]))
    X = np.array(X)
    y = np.array(y)
    print(y)
    return X, y


X, y = get_data()

# print(X)
# print(y)
show_saliency_maps(X, y)
