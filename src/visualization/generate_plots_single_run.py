import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization_utils import load_csv_into_dataframe, generate_accuracy_multiple_plot, generate_gradcam_amp_plot, generate_pct_gutout_pixels_plot, generate_pct_gutout_pixels_n_gradamp_plot



if __name__ == "__main__":

    save_dir = r"results\cifar10\single_run"

    experiment_string_model = "gutout"
    df_model = load_csv_into_dataframe(
        r"C:\Users\benjy\Desktop\DL_results\single_model_gutout_thresh_09\cifar10_cutout_resnet18_a.csv"
    )


    # generate_accuracy_plot(df_model, experiment_string_model, save_dir)
    generate_gradcam_amp_plot(df_model, experiment_string_model, save_dir)
    generate_pct_gutout_pixels_plot(df_model, experiment_string_model, save_dir)
    generate_pct_gutout_pixels_n_gradamp_plot(df_model, experiment_string_model, save_dir)


    experiment_string_baseline = "cutout"
    df_baseline = load_csv_into_dataframe(
        r"C:\Users\benjy\Desktop\DL_results\experiments\cifar10_cutout_single_w_old_gutout\cifar10_cutout_resnet18_a.csv"
    )
    for last_50_epochs in [True, False]:
        title = "cutout_vs_gutout_last_50_epochs" if last_50_epochs else "cutout_vs_gutout"
        generate_accuracy_multiple_plot(
            list_of_dfs=[df_model, df_baseline], 
            list_of_experiment_string=[experiment_string_model, experiment_string_baseline], 
            title=title, 
            save_dir=save_dir,
            last_50_epochs=last_50_epochs
        )