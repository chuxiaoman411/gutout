import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization_utils import load_csv_into_dataframe, generate_accuracy_multiple_plot, generate_gradcam_amp_plot, generate_pct_gutout_pixels_plot, generate_pct_gutout_pixels_n_gradamp_plot



if __name__ == "__main__":

    dataset = "cifar10"

    # choose files
    if dataset == "cifar10":
        save_dir = r"results\cifar10\single_run"
        gutout_file_name = r"C:\Users\benjy\Desktop\DL_results\experiments_new_norm\experiments\cifar10_single_thresh_05\cifar10_cutout_resnet18_a.csv"
        cutout_file_name = r"C:\Users\benjy\Desktop\DL_results\original_cutout\cifar10_resnet18.csv"
    else:
        save_dir = r"results\cifar100\single_run"
        gutout_file_name = r"C:\Users\benjy\Desktop\DL_results\experiments_new_norm\experiments\cifar100_single_thresh_06\cifar100_cutout_resnet18_a.csv"
        cutout_file_name = r"C:\Users\benjy\Desktop\DL_results\original_cutout\cifar100_resnet18.csv"


    experiment_string_model = "gutout"
    df_model = load_csv_into_dataframe(gutout_file_name)


    generate_gradcam_amp_plot(df_model, experiment_string_model, save_dir)
    generate_pct_gutout_pixels_plot(df_model, experiment_string_model, save_dir, dataset)
    generate_pct_gutout_pixels_n_gradamp_plot(df_model, experiment_string_model, save_dir)


    experiment_string_baseline = "cutout"
    df_baseline = load_csv_into_dataframe(cutout_file_name)

    for last_50_epochs in [True, False]:
        title = "cutout_vs_gutout_last_50_epochs" if last_50_epochs else "cutout_vs_gutout"
        generate_accuracy_multiple_plot(
            list_of_dfs=[df_model, df_baseline], 
            list_of_experiment_string=[experiment_string_model, experiment_string_baseline], 
            title=title, 
            save_dir=save_dir,
            last_50_epochs=last_50_epochs
        )