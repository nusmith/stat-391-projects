import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Define densities
def f(x):
    if 0.0 <= x <= 1.0:
        return 2.0 * x
    else:
        return 0.0

def g(x):
    if 0 <= x <= 0.5:
        return 4 * x
    elif 0.5 <= x & x <= 1:
        return 4 * (1 - x)
    else:
        return 0


# Read f, g data
f_train_nums = pd.read_csv('hw5-f-train.dat', header=None)
f_valid_nums = pd.read_csv('hw5-f-valid.dat', header=None)
f_train = [float(i) for i in (f_train_nums.values[0])[0].split(" ")]
f_valid = [float(i) for i in (f_valid_nums.values[0])[0].split(" ")]
g_train_nums = pd.read_csv('hw5-g-train.dat', header=None)
g_valid_nums = pd.read_csv('hw5-g-valid.dat', header=None)
g_train = [float(i) for i in (g_train_nums.values[0])[0].split(" ")]
g_valid = [float(i) for i in (g_valid_nums.values[0])[0].split(" ")]

# sort
f_train.sort()
f_valid.sort()
g_train.sort()
g_valid.sort()


# Gaussian Kernel N(0,1)
def k_gauss(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * x ** 2)


# Mixture (of gaussians) density
def fh(x, k, h, D):
    n = len(D)
    k_sum = 0
    for sample in D:
        k_sum += k((x - sample) / h)
    # normalize
    return (1 / (n * h)) * k_sum


# Likelihood of training & validation data
# possible values for h
h_range = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5]
# Likelihood of the train data under fh
Lh_series_train = np.ones(len(h_range))
# Find likelihood of data for all values of h
fh_train = [[fh(x, k_gauss, h0, f_train) for x in f_train] for h0 in h_range]
Lh_series_train = [sum(np.log10(d) for d in row) for row in fh_train]
print(Lh_series_train)
# Visualize the KDEs
plt.plot(f_train, fh_train[0], label="h= 0.001")
plt.plot(f_train, fh_train[1], label="h= 0.002")
plt.plot(f_train, fh_train[2], label="h= 0.005")
plt.plot(f_train, fh_train[3], label="h= 0.01")
plt.plot(f_train, fh_train[4], label="h= 0.05")
plt.plot(f_train, fh_train[5], label="h= 0.1")
plt.plot(f_train, fh_train[6], label="h= 0.5")
plt.legend()
plt.show()

# Likelihood of the validation under fh
# Use for Cross-Validation to find h*
# Lh_series_test = np.ones(len(h_range))
# for i in range(len(h_range)):
#    h = h_range[i]
# # Find likelihood of data for all values of h
#    for d in f_train:
#       Lh_series_test[i] = Lh_series_test[i] * fh(d, k_gauss, h, f_train)
# print(Lh_series_test)


# Plot fh*
h_star = 0.1
x_d = np.arange(-0.5, 1.5, 0.01)
fh_star = [fh(x, k_gauss, h_star, f_train) for x in x_d]
fx = [f(x_i) for x_i in x_d]
# visualize the kernels
plt.plot(x_d, fh_star, label="Estimated density f_h(x) for h = h*")
plt.plot(x_d, fx, label="True density f(x)")
plt.show()
