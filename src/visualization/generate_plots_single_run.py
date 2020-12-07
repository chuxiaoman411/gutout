import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization_utils import load_csv_into_dataframe, generate_accuracy_plot, generate_gradcam_amp_plot, generate_pct_gutout_pixels_plot, generate_pct_gutout_pixels_n_gradamp_plot



if __name__ == "__main__":

    csv_filename = (
        r"C:\Users\benjy\Desktop\DL_results\single_model_gutout_thresh_09\cifar10_cutout_resnet18_a.csv"
    )
    experiment_string = "model_a"
    save_dir = r"results\cifar10\single_run"
    df = load_csv_into_dataframe(csv_filename)


    generate_accuracy_plot(df, experiment_string, save_dir)
    generate_gradcam_amp_plot(df, experiment_string, save_dir)
    generate_pct_gutout_pixels_plot(df, experiment_string, save_dir)
    generate_pct_gutout_pixels_n_gradamp_plot(df, experiment_string, save_dir)