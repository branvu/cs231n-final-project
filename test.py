import pandas as pd

# data = pd.read_csv('annotations.csv')
# print(data)
# print(data['0'])


labels = {}

for i in range(26):
    labels[i] = i

labels[9] = 100000
labels[25] = 100000

for i in range(10, 25):
    labels[i] -= 1

print(labels)
