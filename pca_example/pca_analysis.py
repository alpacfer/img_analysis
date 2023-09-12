import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data
data_dir = ""
file_name = "Wine.csv"
data = pd.read_csv(data_dir + file_name)

# Separate data into features and labels
x = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
print(data)
