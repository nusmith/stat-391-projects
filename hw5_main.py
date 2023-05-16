import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Define densities
def f(x):
    if 0 <= x & x <= 1:
        return 2 * x
    else:
        return 0


def g(x):
    if 0 <= x & x <= 0.5:
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

print("range f:" + str([min(f_train), max(f_train)]))

# Gaussian Kernel N(0,1)
def k_gauss(x):
    return 1/np.sqrt(2*np.pi) * np.exp(-1/2 * x**2)

# Mixture (of gaussians) density
def fh(x, k, h, D):
    n = len(D)
    k_sum = 0
    for sample in D:
        k_sum += k((x-sample) / h)
    # normalize
    return (1 / (n * h)) * k_sum


print(fh(f_train[0], k_gauss, 0.001, f_train))
x_d = np.arange(0, 1, 0.001)
dens = [fh(xstep, k_gauss, 0.1, f_train) for xstep in x_d]
#print(dens)
# visualize the kernels
plt.plot(x_d, dens, label = "Gaussian")
plt.show()

lh = 0
i = 1
# for d in f_train:
#     lh += np.log10(fh(x=d, k=k_gauss, h=0.002, D=f_train))
#     print(i)
#     i += 1
#     print(lh)
# print(lh)

# possible values for h
h_range = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
# Likelihood of the data (training) under fh
# Lh_series_train = np.ones(len(h_range))
# for i in range(len(h_range)):
#     # Find likelihood of data for all values of h
#     for d in f_train:
#         Lh_series_train[i] = Lh_series_train[i] * fh(d, k_gauss, h_range[i], f_train)
# print(Lh_series_train)

# Likelihood of the data (test) under fh
# Lh_series_test = np.ones(len(h_range))
# for i in range(len(h_range)):
#    h = h_range[i]
# Find likelihood of data for all values of h
#    for d in f_train:
#       Lh_series_test[i] = Lh_series_test[i] * fh(d, k_gauss, h, f_train)
# print(Lh_series_test)
