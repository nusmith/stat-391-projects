import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define densities
def f(x):
    if 0.0 <= x <= 1.0:
        return 2.0 * x
    else:
        return 0.0


def g(x):
    if 0 <= x <= 0.5:
        return 4 * x
    elif 0.5 <= x <= 1:
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

# ---------------------------------------------------------------------------------------
# PART (a)
# Estimation of density of f via KDE
# possible values for h
h_range = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]
# KDE for training data
fh_train = [[fh(x, k_gauss, h0, f_train) for x in f_train] for h0 in h_range]
# Visualize the KDEs
plt.plot(f_train, fh_train[0], label="h= 0.001")
plt.plot(f_train, fh_train[1], label="h= 0.002")
plt.plot(f_train, fh_train[2], label="h= 0.005")
plt.plot(f_train, fh_train[3], label="h= 0.01")
plt.plot(f_train, fh_train[4], label="h= 0.02")
plt.plot(f_train, fh_train[5], label="h= 0.05")
plt.plot(f_train, fh_train[6], label="h= 0.1")
plt.plot(f_train, fh_train[7], label="h= 0.5")
plt.legend()
plt.title(label="KDE of Training Data w/ h")
plt.xlabel("x")
plt.ylabel("fh(x)")
plt.show()

# KDE for validation data
fh_valid = [[fh(x, k_gauss, h0, f_train) for x in f_valid] for h0 in h_range]
# Visualize the KDEs
plt.plot(f_valid, fh_valid[0], label="h= 0.001")
plt.plot(f_valid, fh_valid[1], label="h= 0.002")
plt.plot(f_valid, fh_valid[2], label="h= 0.005")
plt.plot(f_valid, fh_valid[3], label="h= 0.01")
plt.plot(f_valid, fh_valid[4], label="h= 0.02")
plt.plot(f_valid, fh_valid[5], label="h= 0.05")
plt.plot(f_valid, fh_valid[6], label="h= 0.1")
plt.plot(f_valid, fh_valid[7], label="h= 0.5")
plt.legend()
plt.title(label="KDE of Validation Data w/ h")
plt.xlabel("x")
plt.ylabel("fh(x)")
plt.show()
# Use for Cross-Validation to find h*
# Log-likelihood of the train data under fh-- find log-likelihood of data for all values of h
lh_series_train = [sum(np.log10(d) for d in row) for row in fh_train]
lh_series_valid = [sum(np.log10(d) for d in row) for row in fh_valid]
h_star_f = h_range[lh_series_valid.index(max(lh_series_valid))]
# Plot log-likelihood
plt.plot(h_range, lh_series_train, label="Training Data")
plt.plot(h_range, lh_series_valid, label="Validation Data")
plt.legend()
plt.title(label="log-Likelihood of Training and Validation Data w/ h")
plt.xlabel("h")
plt.ylabel("l^CV")
plt.show()
# Compare fh* to true density
x_range = np.arange(-0.5, 1.5, 0.1)
plt.plot(x_range, [f(x) for x in x_range], label="f(x) true")
plt.plot(x_range, [fh(x, k_gauss, h_star_f, f_train) for x in x_range], label="fh*(x)")
plt.legend()
plt.title(label="Compare KDE estimate w/ h* and true density f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()


# PART(b)
# ---------------------------------------------------------------------------------------
# Estimation of density of g via KDE
# possible values for h
h_range = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]
# KDE for training data
gh_train = [[fh(x, k_gauss, h0, g_train) for x in g_train] for h0 in h_range]
# Visualize the KDEs
plt.plot(g_train, gh_train[0], label="h= 0.001")
plt.plot(g_train, gh_train[1], label="h= 0.002")
plt.plot(g_train, gh_train[2], label="h= 0.005")
plt.plot(g_train, gh_train[3], label="h= 0.01")
plt.plot(g_train, gh_train[4], label="h= 0.02")
plt.plot(g_train, gh_train[5], label="h= 0.05")
plt.plot(g_train, gh_train[6], label="h= 0.1")
plt.plot(g_train, gh_train[7], label="h= 0.5")
plt.legend()
plt.title(label="KDE of Training Data w/ h")
plt.xlabel("x")
plt.ylabel("gh(x)")
plt.show()

# KDE of validation data
gh_valid = [[fh(x, k_gauss, h0, g_train) for x in g_valid] for h0 in h_range]
# Visualize the KDEs
plt.plot(g_valid, gh_valid[0], label="h= 0.001")
plt.plot(g_valid, gh_valid[1], label="h= 0.002")
plt.plot(g_valid, gh_valid[2], label="h= 0.005")
plt.plot(g_valid, gh_valid[3], label="h= 0.01")
plt.plot(g_valid, gh_valid[4], label="h= 0.02")
plt.plot(g_valid, gh_valid[5], label="h= 0.05")
plt.plot(g_valid, gh_valid[6], label="h= 0.1")
plt.plot(g_valid, gh_valid[7], label="h= 0.5")
plt.legend()
plt.title(label="KDE of Validation Data w/ h")
plt.xlabel("x")
plt.ylabel("gh(x)")
plt.show()
# Use for Cross-Validation to find h*
# Likelihood of the train data under fh-- find likelihood of data for all values of h
lh_series_train = [sum(np.log10(d) for d in row) for row in gh_train]
lh_series_valid = [sum(np.log10(d) for d in row) for row in gh_valid]
h_star_g = h_range[lh_series_valid.index(max(lh_series_valid))]
# Plot likelihood
plt.plot(h_range, lh_series_train, label="Training Data")
plt.plot(h_range, lh_series_valid, label="Validation Data")
plt.legend()
plt.title(label="log-Likelihood of Training and Validation Data w/ h")
plt.xlabel("h")
plt.ylabel("l^CV")
plt.show()
# Compare fh* to true density
x_range = np.arange(-0.5, 1.5, 0.1)
plt.plot(x_range, [g(x) for x in x_range], label="g(x) true")
plt.plot(x_range, [fh(x, k_gauss, h_star_g, g_train) for x in x_range], label="gh*(x)")
plt.legend()
plt.title(label="Compare KDE estimate w/ h* and true density g(x)")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.show()
# -------------------------------------------------------------------------------------------
# PART (c)
# See below that h_star_f = 0.01 and h_star_g = 0.02
print("h*_f = " + str(h_star_f), "h*_g = " + str(h_star_g))

