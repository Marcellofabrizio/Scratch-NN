import numpy as np
import pandas as pd

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
data = pd.read_csv('../data/train.csv', ',')
data['CLASS'] = [mapping[item] for item in data['CLASS']]

test_data = pd.read_csv('../data/test.csv', ',')
test_data['CLASS'] = [mapping[item] for item in test_data['CLASS']]


nn = Network([18, 13, 7], 0.01, 0.01)

positives = 0
all_cases = 0
positives_train = 0
all_cases_train = 0

data = data.to_numpy()
test_data = test_data.to_numpy()
print(data[0])
for _ in range(200):
    for sample in data:
        result = nn.feedforward(sample[1:])
        nn.backprop(int(sample[0]))

        if result == int(sample[0]):
            positives_train += 1

        all_cases_train += 1
        
print(positives_train)
print(all_cases_train)

for sample in test_data:
    result = nn.feedforward(sample[1:])
    expected = int(sample[0])
    if result == expected:
        positives += 1
    all_cases += 1
        
print(positives)
print(all_cases)