import numpy as np

def print_metrics(confusion_matrix, class_mapping):
    labels = list(class_mapping.keys())

    print_confusion_matrix(confusion_matrix, labels)
    print()
    print_precision_and_recall(confusion_matrix, labels)
    print()
    print(f'Acurácia do modelo: {accuracy(confusion_matrix)*100:.2f}%')


def print_confusion_matrix(confusion_matrix, labels):

    padding = len(labels[0]) + 2
    label_top = ''
    for l in labels:
        label_top += l.ljust(padding)

    print(label_top.rjust(len(label_top) + padding))

    for i in range(len(confusion_matrix)):
        label = labels[i]
        line = label.ljust(padding)
        for j in range(len(confusion_matrix[i])):
            line += f'{confusion_matrix[i, j]}'.ljust(padding)
        print(line)


def accuracy(confusion_matrix):

    true_positives = np.trace(confusion_matrix)
    all_elements = np.sum(confusion_matrix)

    return true_positives/all_elements


def print_precision_and_recall(confusion_matrix, labels):

    label_padding = len(labels[0]) + 2
    labels_top = ['Precision', 'Recall', 'F1-Score']
    labels_top_padding = len(labels_top[0])+2
    label_top = ''

    for l in labels_top:
        label_top += l.ljust(labels_top_padding)

    print(label_top.rjust(len(label_top) + label_padding))

    for i in range(len(labels)):
        label = labels[i]
        line = label.ljust(label_padding)

        precision = class_precision(confusion_matrix, i)
        recall = class_recall(confusion_matrix, i)
        f1_score = class_f1_score(precision, recall)

        line += f'{precision:.2f}'.ljust(labels_top_padding)
        line += f'{recall:.2f}'.ljust(labels_top_padding)
        line += f'{f1_score:.2f}'.ljust(labels_top_padding)

        print(line)


def class_recall(confusion_matrix, class_label):
    true_positives = confusion_matrix[class_label, class_label]
    tp_fn = np.sum(confusion_matrix[class_label])

    return true_positives/tp_fn


def class_precision(confusion_matrix, class_label):
    matrix_col_sum = np.sum(confusion_matrix, axis=0)
    true_positives = confusion_matrix[class_label, class_label]
    tp_fp = matrix_col_sum[class_label]

    return true_positives/tp_fp


def class_f1_score(precision, recall):
    return 2*(recall*precision)/(recall+precision)