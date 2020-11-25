import csv
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_csv_into_dataframe(csv_filename):
    lines_for_temp_csv = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # print(', '.join(row))

            if len(row) > 2:
                lines_for_temp_csv.append(','.join(row) + "\n")

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
    plt.title(f"accuracy_{experiment_string}")
    plt.savefig(os.path.join(save_dir, f"accuracy_{experiment_string}.png"))


if __name__ == "__main__":
    csv_filename = r"25-11-2020__20-57-30_experiment_cifar10_resnet18\cifar10_resnet18_a.csv"
    experiment_string = "model_b"
    save_dir = os.path.dirname(csv_filename)


    df = load_csv_into_dataframe(csv_filename)
    generate_accuracy_plot(df, experiment_string, save_dir)