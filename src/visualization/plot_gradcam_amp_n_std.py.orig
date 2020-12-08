import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from plot_accuracy import load_csv_into_dataframe

def generate_gradcam_amp_plot(df, experiment_string, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.errorbar(df["epoch"], df["gradamp_mean_mean"], yerr=df["gradamp_std_mean"], fmt='-o')

    plt.legend(["mean amplitude of gradcam"])
    plt.xlabel("epochs")
    plt.ylabel("amplitude of gradcam")
    plt.title(f"Mean amplitude of gradcam w error {experiment_string}")
    plt.savefig(os.path.join(save_dir, f"gradamp_w_error_{experiment_string}.png"))


if __name__ == "__main__":
    csv_filename = (
        # r"./../training/test_experiment_cifar10_cutout_resnet18/cifar10_cutout_resnet18_b.csv"
        r"./../training/experiments/recent_experiment_cifar10_cutout_resnet18/cifar10_cutout_resnet18_b.csv"
    )
    experiment_string = "model_a"
    save_dir = os.path.dirname(csv_filename)

    df = load_csv_into_dataframe(csv_filename)
    generate_gradcam_amp_plot(df, experiment_string, save_dir)
