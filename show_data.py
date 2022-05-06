import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

frame = pd.read_csv('annotations.csv')

img_name = frame.iloc[1, 0]
classifications = frame.iloc[1, 1:]
classifications = np.array(classifications)

print('Image name: {}'.format(img_name))
print('Classifications: {}'.format(classifications))

im = cv2.imread(img_name)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()