import pandas as pd
import cv2
from preprocess_functions import *

# data = pd.read_csv('annotations.csv')
# print(data)
# print(data['0'])
img = cv2.imread("hand identifier.png")
cv2.imwrite("hand_cropped.png", grayscale(resize(100, 100, img)))
cv2.waitKey(0)
