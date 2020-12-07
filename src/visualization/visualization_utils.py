import os
import pandas as pd
import matplotlib.pyplot as plt
import csv


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