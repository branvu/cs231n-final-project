from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from mydata import SignLanguageDataset
from models import LanguageModel

'''
MODIFY CONSTANTS AS NEEDED
'''
BATCH_SIZE = 10
EPOCHS = 1
ANNOTATIONS = "annotations/annotations.csv"
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
            letter = data[letter].to(device)

            # zero out gradients
            optimizer.zero_grad()
            out = model(inputs)

            # modify structure of language tensor to match labels
            # language: [0, 0, 1]
            # new_lang: [[1, 0, 0], [1, 0, 0], [0, 1, 0]]
            new_lang = []
            for j in language:
                curr = [0] * 3
                curr[j] = 1
                new_lang.append(curr)
            new_lang = torch.Tensor(new_lang)
            new_lang = new_lang.to(device)

            # calculate loss and perform backwards pass
            loss = lo(out, new_lang)
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
            correct += (predicted == language).sum().item()
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
    setup()

    # create model
    model = LanguageModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Start training")
    print("="*50, "\n")

    train(model, optimizer, epochs=EPOCHS)

    print("Test accuracy:")
    print("="*50)
    check_accuracy(testLoader, model)

    # Saving the model
    torch.save(model.state_dict(), 'models/model_basic.pth')


if __name__ == '__main__':
    main()
