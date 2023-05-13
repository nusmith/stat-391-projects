import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define densities
def f(x):
    if (0 <= x & x <= 1):
        return 2*x
    else:
        return 0

def g(x):
    if (0 <= x & x <= 0.5):
        return 4*x
    elif (0.5 <= x & x <= 1):
        return 4*(1-x)
    else:
        return 0

# Read f, g data
f_train_nums = pd.read_csv('hw5-f-train.dat', header=None)
f_valid_nums = pd.read_csv('hw5-f-valid.dat', header=None)
f_train = [float(i) for i in (f_train_nums.values[0])[0].split(" ")]
f_valid = (f_valid_nums.values[0])[0].split(" ")

# Gaussian Kernel N(0,1)
def K(x):
    return float(1/(np.sqrt(2*np.pi))) * (np.e ** ((-1/2) * x ** 2))

# print(K(4))
# plt.plot(np.arange(4 - 5, 4 + 5, 0.01), K(4))
# plt.show()
# Mixture (of gaussians) density
def fh(x, k, h):
    n = len(f_train)
    k_sum = 0
    for t in f_train:
        k_sum += k((x - t) / h)
    # normalize
    return (1 / (n * h)) * k_sum


# possible values for h
h_range = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
# Likelihood of the data (training) under fh
Lh_series_train = np.ones(len(h_range))
for i in range(len(h_range)):
    h = h_range[i]
    # Find likelihood of data for all values of h
    for d in f_train:
        Lh_series_train[i] = Lh_series_train[i] * fh(d, K, h)
print(Lh_series_train)

# Likelihood of the data (test) under fh
Lh_series_test = np.ones(len(h_range))
for i in range(len(h_range)):
    h = h_range[i]
    # Find likelihood of data for all values of h
    for d in f_train:
        Lh_series_test[i] = Lh_series_test[i] * fh(d, K, h)
print(Lh_series_test)








