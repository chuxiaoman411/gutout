import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from plot_accuracy import load_csv_into_dataframe

def generate_pct_gutout_pixels_plot(df, experiment_string, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.errorbar(df["epoch"], df["train_num_masked_pixel"]/1024.0, yerr=df["gutout_std_mean"]/1024.0, fmt='-o')

    plt.legend(["mean fraction of gutout pixels"])
    plt.xlabel("epochs")
    plt.ylabel("fraction of gutout pixels")
    plt.title(f"Mean fraction of gutout pixels w error {experiment_string}")
    plt.savefig(os.path.join(save_dir, f"pct_gutout_pix_w_error_{experiment_string}.png"))


if __name__ == "__main__":
    csv_filename = (
        r"./../training/experiments/recent_experiment_cifar10_cutout_resnet18/cifar10_cutout_resnet18_b.csv"
    )
    experiment_string = "model_b"
    save_dir = os.path.dirname(csv_filename)

    df = load_csv_into_dataframe(csv_filename)
    generate_pct_gutout_pixels_plot(df, experiment_string, save_dir)
