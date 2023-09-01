import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
in_dir = ""
txt_name = "irisdata.txt"
iris_data = np.loadtxt(in_dir + txt_name, comments="%")
x = iris_data[0:50, 0:4]  # x is a matrix with 50 rows and 4 columns
n_samples = x.shape[0]  # number of samples

# Show data in a plot
plt.figure()
# Transform data to a pandas dataframe
d = pd.DataFrame(x, columns=["Sepal length", "Sepal width", "Petal length", "Petal width"])
sns.pairplot(d)
plt.show()
# Conclusion: sepal length and petal length are less correlated than sepal length and sepal width

# PCA analysis
# Center data
mn = np.mean(x, axis=0)
data = x - mn
# Covariance matrix
cov_1 = 1 / (n_samples - 1) * np.matmul(data.T, data)
cov_2 = np.cov(data.T)
cov = cov_1
# Eigenvector analysis
eig_val, eig_vec = np.linalg.eig(cov)
eig_val_norm = eig_val / np.sum(eig_val)
# Percentage explained variance for first principal component
print("% explainded first component: ", eig_val_norm[0])
# Plot eigenvalues
plt.figure()
plt.plot(eig_val_norm, "o")
plt.xlabel("Principal component")
plt.ylabel("Percent explained variance")
plt.ylim([0, 1])
plt.show()

# Project data on first principal component
