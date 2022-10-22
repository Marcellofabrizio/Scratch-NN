import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from network import Network
from metrics import print_metrics

mapping = {
    'BRICKFACE': 0,
    'SKY': 1,
    'FOLIAGE': 2,
    'CEMENT': 3,
    'WINDOW': 4,
    'PATH': 5,
    'GRASS': 6
}

train_data = pd.read_csv(
    '/home/marcello/Repositories/scratch-neural-net/data/train.csv', ',')
train_data['CLASS'] = [mapping[item] for item in train_data['CLASS']]

test_data = pd.read_csv(
    '/home/marcello/Repositories/scratch-neural-net/data/test.csv', ',')
test_data['CLASS'] = [mapping[item] for item in test_data['CLASS']]


nn = Network([18, 14, 7], 1, 1)

train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

cf_matrix = np.zeros((7, 7))

for epoch in range(100):
    for sample in train_data:
        expected_output = int(sample[0])
        result = nn.feedforward(sample[1:])
        nn.backprop(expected_output)

for sample in test_data:
    expected_output = int(sample[0])
    result = nn.feedforward(sample[1:])

    cf_matrix[expected_output, result] += 1

cf_matrix = cf_matrix.astype(int)

print_metrics(cf_matrix, mapping)

# confusion_matrix = np.array([[29.,  0.,  0.,  1.,  0.,  0.,  0.],
#                              [0., 30.,  0.,  0.,  0.,  0.,  0.],
#                              [0.,  0., 18.,  1., 11.,  0.,  0.],
#                              [0.,  0.,  0., 29.,  1.,  0.,  0.],
#                              [0.,  0.,  0.,  4., 26.,  0.,  0.],
#                              [3.,  0.,  0.,  0.,  0., 27.,  0.],
#                              [0.,  0.,  0.,  0.,  0.,  0., 30.]])
