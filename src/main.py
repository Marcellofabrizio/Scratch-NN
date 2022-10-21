import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from network import Network

mapping = {
    'BRICKFACE': 0,
    'SKY': 1,
    'FOLIAGE': 2,
    'CEMENT': 3,
    'WINDOW': 4,
    'PATH': 5,
    'GRASS': 6
}

def print_confusion_matrix():
    confusion_matrix = np.array([[29.,  0.,  0.,  1.,  0.,  0.,  0.],
                                 [0., 30.,  0.,  0.,  0.,  0.,  0.],
                                 [0.,  0., 18.,  1., 11.,  0.,  0.],
                                 [0.,  0.,  0., 29.,  1.,  0.,  0.],
                                 [0.,  0.,  0.,  4., 26.,  0.,  0.],
                                 [3.,  0.,  0.,  0.,  0., 27.,  0.],
                                 [0.,  0.,  0.,  0.,  0.,  0., 30.]])

    labels = list(mapping.keys())
    padding = len(labels[0]) + 1

    for i in range(len(confusion_matrix)):
        label = labels[i]
        print(label.ljust(padding), confusion_matrix[i])
        # for j in range(len(confusion_matrix[i])):
        #     str(confusion_matrix[i, j].astype(int))


print_confusion_matrix()
exit(1)

train_data = pd.read_csv(
    '/home/marcello/Repositories/Scratch-NN/data/train.csv', ',')
train_data['CLASS'] = [mapping[item] for item in train_data['CLASS']]

test_data = pd.read_csv(
    '/home/marcello/Repositories/Scratch-NN/data/test.csv', ',')
test_data['CLASS'] = [mapping[item] for item in test_data['CLASS']]


nn = Network([18, 14, 7], 1, 1)

positives = 0
all_cases = 0
positives_train = 0
all_cases_train = 0

train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

cf_matrix = np.zeros((7, 7))

for epoch in range(200):
    for sample in train_data:
        expected_output = int(sample[0])
        result = nn.feedforward(sample[1:])
        nn.backprop(expected_output)

        if result == expected_output:
            positives_train += 1
        all_cases_train += 1

print(positives_train)
print(all_cases_train)

for sample in test_data:
    expected_output = int(sample[0])
    result = nn.feedforward(sample[1:])

    if result == expected_output:
        positives += 1
    all_cases += 1

    cf_matrix[expected_output, result] += 1

print(positives)
print(all_cases)
print(cf_matrix)
