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
# Gaussian Kernel centered at x
def K(x):
    mu = x
    s = 1
    # kernel width will be 5 units
    x_range = np.arange(mu - 5, mu + 5, 0.01)
    x_range_norm = [(i - mu)/s for i in x_range]
    return [float(1/(np.sqrt(2*np.pi)*s)) * (np.e**((-1/2)*(x_i)**2)) for x_i in x_range_norm]

# print(K(4))
# plt.plot(np.arange(4 - 5, 4 + 5, 0.01), K(4))
# plt.show()
# Mixture (of gaussians) density
def fh(x, k, h):
    n = len(f_train)
    return (1/(n*h)) * sum([k(-1*(sample - x)/h) for sample in f_train])


print(fh(4, K, 0.01))


# possible values for h
h_range = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

# Likelihood of the data (training) under fh
L_series_train = np.ones(len(h_range))
# for i in range(1, len(h_range) + 1):
#     h = h_range[i]
#     for d in f_train:
#         break
#         L_series_train[i] = L_series_train[i] * fh()


# Likelihood of the data (test) under fh
L_series_test = np.ones(len(h_range))









