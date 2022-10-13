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
data = pd.read_csv('../data/dados-train.csv', ',')
data['CLASS'] = [mapping[item] for item in data['CLASS']]

nn = Network([19, 13, 6], 0.2, 0.9)

positives = 0
all_cases = 0

data = data.to_numpy()
for _ in range(100):
    for sample in data:
        result = nn.feedforward(sample)
        nn.backprop(result)
        expected = sample[0]
        if result == expected:
            positives += 1
        
        all_cases += 1

print(positives)
print(all_cases)