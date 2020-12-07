import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np


def load_csv_into_dataframe(csv_filename):

    # load csv and remove header
    lines_for_temp_csv = []
    with open(csv_filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) > 2:
                lines_for_temp_csv.append(",".join(row) + "\n")

    # 
    with open("temp.csv", "w") as csvfile:
        for line in lines_for_temp_csv:
            csvfile.write(line)

    df = pd.read_csv("temp.csv")
    os.remove("temp.csv")
    return df


def generate_accuracy_plot(df, experiment_string, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.plot(df["epoch"], df["train_acc"], "b--")
    plt.plot(df["epoch"], df["test_acc"], "b")

    plt.legend(["train accuracy", "test accuracy"])
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(f"accuracy - {experiment_string}")
    plt.savefig(os.path.join(save_dir, f"accuracy_{experiment_string}.png"))
    plt.close()


def generate_gradcam_amp_plot(df, experiment_string, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.errorbar(df["epoch"], df["gradamp_mean_mean"], yerr=[np.zeros(len(df["gutout_std_mean"])), df["gutout_std_mean"]/1024.0], fmt='-o')

    plt.legend(["mean amplitude of gradcam"])
    plt.xlabel("epochs")
    plt.ylabel("amplitude of gradcam")
    plt.title(f"Amplitude of gradcam - {experiment_string}")
    plt.savefig(os.path.join(save_dir, f"gradamp_amp_{experiment_string}.png"))
    plt.close()


def generate_pct_gutout_pixels_plot(df, experiment_string, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # plt.errorbar(df["epoch"], df["train_num_masked_pixel"]/1024.0, yerr=[np.zeros(len(df["gutout_std_mean"])), df["gutout_std_mean"]/1024.0], fmt='-o')
    # plt.plot(df["epoch"], df["train_num_masked_pixel"]/1024.0, "-k", linewidth=5)
    plt.errorbar(df["epoch"], df["train_num_masked_pixel"]/1024.0, yerr=[np.zeros(len(df["gutout_std_mean"])), df["gutout_std_mean"]/1024.0], color="grey")
    plt.plot(df["epoch"], df["train_num_masked_pixel"]/1024.0, color="blue", linewidth=5)

    plt.legend(["mean fraction of gutout pixels", "std fraction of gutout pixels"])
    plt.xlabel("epochs")
    plt.ylabel("fraction of gutout pixels")
    plt.title(f"fraction of gutout pixels - {experiment_string}")
    plt.savefig(os.path.join(save_dir, f"num_gut_pix{experiment_string}.png"))
    plt.close()


def generate_pct_gutout_pixels_n_gradamp_plot(df, experiment_string, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    lns1 = ax1.plot(df["epoch"], df["train_num_masked_pixel"]/1024.0, "b")
    lns2 = ax2.plot(df["epoch"], df["gradamp_mean_mean"], "g")

    # ax1.set_ylim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax1.set_ylabel("fraction of gutout pixels")
    ax2.set_ylabel("gradcam amplitude")
    plt.xlabel("epochs")
    ax1.set_xlabel("epochs")

    lns = lns1 + lns2
    legends = ["fraction of gutout pixels", "gradcam amplitude"]
    ax1.legend(lns, legends)

    plt.title(f"Fraction of gutout pixels & amplitude of gradcam - {experiment_string}")
    plt.savefig(os.path.join(save_dir, f"num_gut_pix_and_grad_amp_{experiment_string}.png"))
    plt.close()