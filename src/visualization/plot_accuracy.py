import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization_utils import load_csv_into_dataframe

def generate_accuracy_plot(df, experiment_string, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.plot(df["epoch"], df["train_acc"], "b--")
    plt.plot(df["epoch"], df["test_acc"], "b")

    plt.legend(["train accuracy", "test accuracy"])
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(f"accuracy_{experiment_string}")
    plt.savefig(os.path.join(save_dir, f"accuracy_{experiment_string}.png"))


if __name__ == "__main__":
    csv_filename = (
        # r"./../training/jerryAB(0.001LR)_experiment_cifar10_cutout_resnet18/cifar10_cutout_resnet18_b.csv"
        r"05-12-2020__07-57-58_experiment_cifar10_cutout_resnet18\cifar10_cutout_resnet18_a.csv"
    )
    
    experiment_string = "model_a"
    save_dir = os.path.dirname(csv_filename)

    df = load_csv_into_dataframe(csv_filename)
    generate_accuracy_plot(df, experiment_string, save_dir)
