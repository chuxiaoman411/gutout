import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from visualization_utils import load_csv_into_dataframe, generate_accuracy_multiple_plot, generate_gradcam_amp_plot, generate_pct_gutout_pixels_plot, generate_pct_gutout_pixels_n_gradamp_plot



if __name__ == "__main__":

    save_dir = r"results\cifar10\joint_training"
    # cutout_file_name = r"C:\Users\benjy\Desktop\DL_results\original_cutout\cifar10_resnet18.csv"

    dataset = "cifar10"

    if dataset == "cifar10":
        save_dir = r"results\cifar10\joint_training"
        model_a_path = r"C:\Users\benjy\Desktop\DL_results\experiments_new_norm\experiments\cifar10_joint_thresh_06\cifar10_cutout_resnet18_a.csv"
        model_b_path = r"C:\Users\benjy\Desktop\DL_results\experiments_new_norm\experiments\cifar10_joint_thresh_06\cifar10_cutout_resnet18_b.csv"
        joint_path = r"C:\Users\benjy\Desktop\DL_results\experiments_new_norm\experiments\cifar10_joint_thresh_06\cifar10_cutout_resnet18_joint.csv"
        cutout_file_name = r"C:\Users\benjy\Desktop\DL_results\original_cutout\cifar10_resnet18.csv"
    else:
        save_dir = r"results\cifar100\joint_training"
        # gutout_file_name = r"C:\Users\benjy\Desktop\DL_results\experiments_new_norm\experiments\cifar100_single_thresh_06\cifar100_cutout_resnet18_a.csv"
        model_a_path = r"C:\Users\benjy\Desktop\DL_results\experiments\train_gutout_joint\cifar10_cutout_resnet18_a.csv"
        model_b_path = r"C:\Users\benjy\Desktop\DL_results\experiments\train_gutout_joint\cifar10_cutout_resnet18_b.csv"
        joint_path = r"C:\Users\benjy\Desktop\DL_results\experiments\train_gutout_joint\cifar10_cutout_resnet18_joint.csv"
        cutout_file_name = r"C:\Users\benjy\Desktop\DL_results\original_cutout\cifar100_resnet18.csv"


    # generate plots for model a
    experiment_string_a = "model_a"
    df_model_a = load_csv_into_dataframe(model_a_path)
    #     r"C:\Users\benjy\Desktop\DL_results\experiments\train_gutout_joint\cifar10_cutout_resnet18_a.csv"
    # )

    df_model_a["epoch"] -= 1
    df_model_a["epoch"] /= 2
    generate_gradcam_amp_plot(df_model_a, experiment_string_a, save_dir)
    generate_pct_gutout_pixels_plot(df_model_a, experiment_string_a, save_dir, dataset)
    generate_pct_gutout_pixels_n_gradamp_plot(df_model_a, experiment_string_a, save_dir)

    # generate plots for model b
    experiment_string_b = "model_b"
    df_model_b = load_csv_into_dataframe(model_b_path)
    #     r"C:\Users\benjy\Desktop\DL_results\experiments\train_gutout_joint\cifar10_cutout_resnet18_b.csv"
    # )
    df_model_b["epoch"] /= 2
    generate_gradcam_amp_plot(df_model_b, experiment_string_b, save_dir)
    generate_pct_gutout_pixels_plot(df_model_b, experiment_string_b, save_dir, dataset)
    generate_pct_gutout_pixels_n_gradamp_plot(df_model_b, experiment_string_b, save_dir)


    experiment_string_joint = "joint"
    df_joint = load_csv_into_dataframe(joint_path)
    #     r"C:\Users\benjy\Desktop\DL_results\experiments\train_gutout_joint\cifar10_cutout_resnet18_joint.csv"
    # )
    df_joint["epoch"] /= 2

    experiment_string_baseline = "cutout"
    df_baseline = load_csv_into_dataframe(cutout_file_name)


    for last_50_epochs in [True, False]:
        title = "joint_training_last_50_epochs" if last_50_epochs else "joint_training"
        generate_accuracy_multiple_plot(
            list_of_dfs=[df_model_a, df_model_b, df_joint, df_baseline], 
            list_of_experiment_string=[experiment_string_a, experiment_string_b, experiment_string_joint, experiment_string_baseline], 
            title=title, 
            save_dir=save_dir,
            last_50_epochs=last_50_epochs
        )