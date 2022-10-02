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
data = pd.read_csv('../data/segmentation.data', ',')
data['LABEL'] = [mapping[item] for item in data['LABEL']]

nn = Network([20, 13, 6])

np_data = data.to_numpy()
for sample in np_data[:1]:
    result = nn.feedforward(sample)
    z = []
    # DEBUG
    for output in result:
        max_id = np.argmax(output)
        z.append(output[max_id])

    max = np.unravel_index(result.argmax(), result.shape)
    z = np.array(z)
    error = z*(1-z)*(sample[0].astype(int)-z)
    nn.backprop(error)
