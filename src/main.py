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
train_data = pd.read_csv('/home/marcello/Repositories/scratch-neural-net/data/train.csv', ',')
train_data['CLASS'] = [mapping[item] for item in train_data['CLASS']]

test_data = pd.read_csv('/home/marcello/Repositories/scratch-neural-net/data/test.csv', ',')
test_data['CLASS'] = [mapping[item] for item in test_data['CLASS']]


nn = Network([18, 14, 7], 1, 1)

positives = 0
all_cases = 0
positives_train = 0
all_cases_train = 0

train_data = train_data.to_numpy()
test_data = test_data.to_numpy()
print(train_data[0])

cf_matrix = np.zeros((7,7))

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

# def print_confusion_matrix(confusion_matrix, class_dict):
#     for i in range(len(confusion_matrix)):
#         print(confusion_matrix[i])
#         for j in range(len(confusion_matrix[i])):