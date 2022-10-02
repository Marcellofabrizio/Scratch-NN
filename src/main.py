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
data = pd.read_csv('../data/segmentation.test', ',')
data['LABEL'] = [mapping[item] for item in data['LABEL']]

nn = Network([20, 13, 6])

np_data = data.to_numpy()
positives = 0
all_cases = 0
for epoch in range(3000):
    for sample in np_data:
        result = nn.feedforward(sample)
        prediction = np.argmax(result)
        expected = sample[0].astype(int)
        error = result*(1-result)*(expected-result)
        nn.backprop(error)
        if prediction == expected:
            positives += 1
        all_cases += 1

print("Positives", positives)
print("All cases", all_cases)