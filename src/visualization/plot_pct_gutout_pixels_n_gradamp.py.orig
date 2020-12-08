import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from plot_accuracy import load_csv_into_dataframe

def generate_pct_gutout_pixels_n_gradamp_plot(df, experiment_string, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    lns1 = ax1.plot(df["epoch"], df["train_num_masked_pixel"]/1024.0, "b")
    lns2 = ax2.plot(df["epoch"], df["gradamp_mean_mean"], "g")

    plt.xlabel("epochs")
    ax1.set_ylim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax1.set_ylabel("fraction of gutout pixels")
    ax2.set_ylabel("gradcam amplitude")

    lns = lns1 + lns2
    legends = ["fraction of gutout pixels", "gradcam amplitude"]
    ax1.legend(lns, legends)

    plt.title(f"Fraction of gutout pixels & avg gradcam amp {experiment_string}")
    plt.savefig(os.path.join(save_dir, f"pct_gut_pix_n_avg_gradamp_{experiment_string}.png"))


if __name__ == "__main__":
    csv_filename = (
        r"./../training/experiments/recent_experiment_cifar10_cutout_resnet18/cifar10_cutout_resnet18_b.csv"
    )
    experiment_string = "model_b"
    save_dir = os.path.dirname(csv_filename)

    df = load_csv_into_dataframe(csv_filename)
    generate_pct_gutout_pixels_n_gradamp_plot(df, experiment_string, save_dir)
