import torch
import numpy as np
import matplotlib.pyplot as plt
from models import *
from preprocess_functions import *

from builtins import range

from math import sqrt, ceil
import numpy as np

NUM_CHARS = 26 + 41 + 26 + 32
MODEL_SAVE_PATH = "models/model_combo.pth"
image_file = "data/asl/asl-5-3629.jpg"
model = ComboModel(NUM_CHARS)


def visualize_grid(Xs, rows, cols, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * rows + padding * (rows - 1)
    grid_width = W * cols + padding * (cols - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(rows):
        x0, x1 = 0, W
        for x in range(cols):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A * H + A, A * W + A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y * H + y: (y + 1) * H + y, x * W + x: (x + 1) * W + x, :] = Xs[
                    n, :, :, :
                ]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def vis_nn(rows):
    """ visualize array of arrays of images """
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N * H + N, D * W + D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y * H + y: (y + 1) * H + y, x * W + x: (x + 1)
              * W + x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def show_net_weights(W, rows, cols):
    W1 = W.transpose(1, 2, 3, 0)
    im_rgb = cv2.cvtColor(visualize_grid(
        W1, rows, cols, padding=3).astype('uint8'), cv2.COLOR_BGR2RGB)

    plt.imshow(im_rgb)
    plt.gca().axis('off')
    plt.show()


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def show_weights():
    show_net_weights(np.array(activation['c1']), 16, 2)
    show_net_weights(np.array(activation['c2']), 16, 4)
    show_net_weights(np.array(activation['c3']), 16, 4)


checkpoint = torch.load(MODEL_SAVE_PATH)
model.load_state_dict(checkpoint)

x = resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE,
           cv2.imread(image_file, cv2.IMREAD_GRAYSCALE))
x = x[None, :, :, None]
x = torch.Tensor(x)

model.c1.register_forward_hook(get_activation('c1'))
model.c2.register_forward_hook(get_activation('c2'))
model.c3.register_forward_hook(get_activation('c3'))

output = model(x)

show_weights()
