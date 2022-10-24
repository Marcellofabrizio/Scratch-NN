import threading
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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


def execute(train_data, test_data, mapping, nn_layers, learning_rate, momentum, epochs):

    nn = Network(nn_layers, learning_rate, momentum)
    cf_matrix = np.zeros((7, 7))
    errors = list()
    epoch_list = list()

    for epoch in range(epochs):
        for sample in train_data:
            expected_output = int(sample[0])
            result = nn.feedforward(sample[1:])
            nn.backprop(expected_output)

        errors.append(np.sum(np.absolute(nn.output_layer.error)))
        epoch_list.append(epoch)
    print(errors)
    plt.plot(epoch_list, errors)
    plt.xlabel("Épocas")
    plt.ylabel("Erro médio")
    plt.savefig(
        f"../plots/mean_error_{epochs}_epochs_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")

    for sample in test_data:
        expected_output = int(sample[0])
        result = nn.feedforward(sample[1:])

        cf_matrix[expected_output, result] += 1

    cf_matrix = cf_matrix.astype(int)

    print(f"Resultados para treinamento com {epochs} épocas")
    print_metrics(cf_matrix, mapping)


if __name__ == '__main__':

    threads = list()

    nn_layers = [18, 14, 7]
    epochs = [100, 200, 500, 600]
    leanlearning_rate = 1
    momentum = 1

    train_data = pd.read_csv(
        '/home/marcello/Repositories/scratch-neural-net/data/train.csv', ',')
    train_data['CLASS'] = [mapping[item] for item in train_data['CLASS']]

    test_data = pd.read_csv(
        '/home/marcello/Repositories/scratch-neural-net/data/test.csv', ',')
    test_data['CLASS'] = [mapping[item] for item in test_data['CLASS']]

    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()

    execute(train_data, test_data, mapping, [18, 14, 7], 0.2, 1, 200)
    # for i in range(4):

    #     logging.info(f"Main: criando e iniciando thread {i}")
    #     x = threading.Thread(target=execute, args=(
    #         train_data, test_data, mapping, [18, 14, 7], 1, 1, epochs[i],))
    #     threads.append(x)
    #     x.start()

    # for i, thread in enumerate(threads):
    #     logging.info(f"Main: Encerrando thread {i}")
    #     thread.join()
    #     logging.info(f"Main: Thread {i} encerrada")
