import numpy as np

# Import data
in_dir = ""
txt_name = "irisdata.txt"
iris_data = np.loadtxt(in_dir + txt_name, comments="%")
x = iris_data[0:50, 0:4]  # x is a matrix with 50 rows and 4 columns

# Data dimensions
n_features = x.shape[1]  # number of features
n_samples = x.shape[0]  # number of samples

# Vector of features
sep_l = x[:, 0]  # sepal length
sep_w = x[:, 1]  # sepal width
pet_l = x[:, 2]  # petal length
pet_w = x[:, 3]  # petal width

# Variance of features: ddof=1 for unbiased estimate
sep_l_var = sep_l.var(ddof=1)
sep_w_var = sep_w.var(ddof=1)
pet_l_var = pet_l.var(ddof=1)
pet_w_var = pet_w.var(ddof=1)

# Covariance between sepal length and sepal width
sep_l_w_cov = 1 / (n_samples - 1) * np.dot(sep_l - sep_l.mean(), sep_w - sep_w.mean())
print("Cov sepal length and width: ", sep_l_w_cov)

# Covariance between sepal length and petal length
sep_l_pet_l_cov = 1 / (n_samples - 1) * np.dot(sep_l - sep_l.mean(), pet_l - pet_l.mean())
print("Cov sepal length and petal length: ", sep_l_pet_l_cov)

# Conclusion: sepal length and petal length are more correlated than sepal length and sepal width
