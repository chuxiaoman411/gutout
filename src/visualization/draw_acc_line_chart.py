import matplotlib.pyplot as plt
import numpy as np
def draw_chart(epochs,accs,labels):
    for i in range(len(labels)):
        e = len(epochs)
        label = str(round(labels[i],3))
        acc = accs[e*i:e*i+e]
        plt.plot(epochs,acc,label=label)

    plt.title('Rigorously Tuning Threshold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.show()


def draw_chart2(epochs, accs, thresholds):
    for i in range(len(epochs)):
        print(len(epochs))
        print(len(accs))
        print(len(thresholds))
        e = len(epochs)
        label = "epoch:"+str(round(epochs[i], 3))
        acc = []
        for j in range(i, len(accs), 6):
            acc.append(accs[j])
        plt.plot(thresholds, acc, label=label)

    plt.title('Rigorously Tuning Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.show()


accs = [0.346,
        0.47,
        0.683,
        0.767,
        0.785,
        0.795,
        0.388,
        0.54,
        0.732,
        0.749,
        0.785,
        0.811,
        0.392,
        0.57,
        0.725,
        0.784,
        0.775,
        0.811,
        0.387,
        0.527,
        0.748,
        0.796,
        0.799,
        0.815,
        0.392,
        0.585,
        0.708,
        0.802,
        0.783,
        0.774,
        0.246,
        0.412,
        0.563,
        0.757,
        0.806,
        0.793,
        0.414,
        0.435,
        0.732,
        0.788,
        0.783,
        0.8,
        0.409,
        0.442,
        0.734,
        0.769,
        0.786,
        0.81,
        0.391,
        0.499,
        0.65,
        0.774,
        0.788,
        0.776,
        0.304,
        0.537,
        0.653,
        0.784,
        0.785,
        0.792,
        0.299,
        0.269,
        0.718,
        0.76,
        0.785,
        0.795,
        0.345,
        0.357,
        0.611,
        0.732,
        0.774,
        0.788,
        0.399,
        0.534,
        0.725,
        0.763,
        0.793,
        0.794,
        0.385,
        0.548,
        0.74,
        0.786,
        0.775,
        0.787,
        0.303,
        0.382,
        0.699,
        0.746,
        0.782,
        0.772,
        0.418,
        0.592,
        0.733,
        0.772,
        0.791,
        0.762,
        0.272,
        0.417,
        0.661,
        0.745,
        0.797,
        0.806,
        0.378,
        0.583,
        0.728,
        0.778,
        0.797,
        0.776,
        0.347,
        0.523,
        0.691,
        0.771,
        0.781,
        0.784,
        0.196,
        0.536,
        0.708,
        0.774,
        0.791,
        0.78,
        0.294,
        0.548,
        0.726,
        0.775,
        0.803,
        0.801]
epochs = list(range(0, 16, 3))
labels = list(np.arange(0.8,0.901,0.005))
draw_chart2(epochs,accs,labels)
