import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from mydata import SignLanguageDataset
from models import ASLModel, JSLModel, ISLModel
import os

'''
MODIFY CONSTANTS AS NEEDED
'''
BATCH_SIZE = 10
EPOCHS = 3
ANNOTATIONS_DIR = "annotations"
NUM_CLASSES = {"asl": 26, "isl": 26, "jsl": 41}

# parse argument of language to train
available_langs = [x.split('_')[0] for x in os.listdir(ANNOTATIONS_DIR)[1:]]
parser = argparse.ArgumentParser()
parser.add_argument(
    "lang", help=f"Specified language to create annotations for. Available languages: {available_langs}")
args = parser.parse_args()

ANNOTATIONS = f"annotations/{args.lang}_annotations.csv"
MODEL_SAVE_PATH = f"models/model_basic_{args.lang}_letter.pth"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trainLoader, valLoader, testLoaer = None, None, None


def setup():
    global trainLoader, valLoader, testLoader

    # Read in dataset from annotations file
    dataset = SignLanguageDataset(ANNOTATIONS, None)
    size = len(dataset)
    splits = [int(size * 0.80), int(size * 0.10), int(size * 0.10) + 1]
    print(f"Splits: {splits}, Data size: {size}")

    assert size == sum(splits)

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, splits)

    # create DataLoaders
    trainLoader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valLoader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testLoader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


def train(model, optimizer, epochs=1):
    lo = nn.CrossEntropyLoss()
    model.to(device)  # Set model to GPU

    for _ in range(epochs):
        for i, data in enumerate(trainLoader, 0):
            model.train()  # set model to train mode

            # move to device (GPU)
            inputs, language, letter = data
            inputs = data[inputs].to(device)
            language = data[language].to(device)

            # letter = data[letter]

            # print(type(letter), letter)
            # new_letter = []
            # if args.lang != "asl":
            #     for j in letter:
            #         new_letter.append(j.item() % 100)
            #     # letter = letter[1:]
            #     letter = new_letter
            letter = data[letter].to(device)

            # zero out gradients
            optimizer.zero_grad()
            out = model(inputs)

            # calculate loss and perform backwards pass
            loss = lo(out, letter)
            loss.backward()

            # update params
            optimizer.step()

            if i % 200 == 0:
                print('Iteration %d, loss = %.4f' % (i, loss.item()))
                check_accuracy(valLoader, model)
                print()


def check_accuracy(loader, model):
    # if loader.dataset.train:
    #     print('Checking accuracy on validation set')
    # else:
    #     print('Checking accuracy on test set')

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            # move to device
            inputs, language, letter = data
            inputs = data[inputs].to(device)
            language = data[language].to(device)
            letter = data[letter].to(device)

            out = model(inputs)
            _, predicted = torch.max(out.data, 1)
            correct += (predicted == letter).sum().item()
            total += predicted.size(0)
        acc = float(correct) / total
        print('Got %d / %d correct (%.2f)' % (correct, total, 100 * acc))


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    '''
    # The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    '''
    print(f"Training model on {args.lang} dataset")
    setup()

    # create model
    if args.lang == "asl":
        model = ASLModel()
    elif args.lang == "jsl":
        print("here")
        model = JSLModel()
    elif args.lang == "isl":
        model = ISLModel()
    else:
        raise Exception(f"Invalid language {args.lang}")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Start training")
    print("="*50, "\n")

    train(model, optimizer, epochs=EPOCHS)

    print("Test accuracy:")
    print("="*50)
    check_accuracy(testLoader, model)

    # Saving the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()
