import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define distributions
def gamma(x):
    return np.math.gamma(x)


# Dirichlet/ Beta prior with alpha1 and alpha2 data
def Diri(theta1, theta2, alpha1, alpha2):
    return gamma(alpha1 + alpha2) / (gamma(alpha1) * gamma(alpha2)) * \
           theta1 ** (alpha1 - 1) * theta2 ** (alpha2 - 1)


# ------------------------------------------------------------------------------------------
# PART (a) change of variable
# Ej = ln(thetaj)
# S = {1, 2}
# Prior
def Diri_E(E1, E2, alpha1, alpha2):
    theta1 = np.e ** E1
    theta2 = np.e ** E2
    return Diri(theta1, theta2, alpha1, alpha2)


# Likelhiood of data n
def L(theta1, theta2, n1, n2):
    return theta1 ** n1 * theta2 ** n2


# Change of variable for Likelihood
def L_E(E1, E2, n1, n2):
    theta1 = np.e ** E1
    theta2 = np.e ** E2
    return theta1 ** n1 * theta2 ** n2


# Calculate posterior, in terms of E, (f ~ f0*L)
def Diri_E_0(E1, E2, n1, n2, alpha1, alpha2):
    (gamma(alpha1 + alpha2) / (gamma(alpha1) * gamma(alpha2))) * \
    L_E(E1, E2, n1, n2) * Diri_E(E1, E2, alpha1, alpha2)


# ------------------------------------------------------------------------------------------
# PART (b) -- given data, analyze posterior distributions for different priors
# Data
D = [1, 1, 2, 1, 1]
theta_range = np.arange(0.01, 1, 0.01)
# Note theta ML
theta1_ML = 0.8
theta2_ML = 0.2
plt.axvline(x=theta2_ML, color="black", label="theta2_ML", linestyle='dashed')
# Plotting prior + posteior distribution of theta2 (theta1 = 1 - theta2)
# PRIORS
# Prior #1: Diri(theta1, theta2; alpha1 = 10, alpha2 = 1)
plt.plot(theta_range, [Diri(1 - theta, theta, 10, 1) for theta in theta_range],
         label="Prior a1=10, a2=1", color="#E9967A")
# Prior #2: Diri(theta1, theta2; alpha1 = 10, alpha2 = 10)
plt.plot(theta_range, [Diri(1 - theta, theta, 10, 10) for theta in theta_range],
         label="Prior a1=10, a2=10", color="#FF7256")
# Prior #3: Diri(theta1, theta2; alpha1 = 0.1, alpha2 = 0.2)
plt.plot(theta_range, [Diri(1 - theta, theta, 0.1, 0.2) for theta in theta_range],
         label="Prior a1=0.1, a2=0.2", color="#8B3E2F")
# POSTERIORS
# Posterior #1: Diri(theta1, theta2; alpha1 = 14, alpha2 = 2)
plt.plot(theta_range, [Diri(1 - theta, theta, 14, 2) for theta in theta_range],
         label="Posterior a1'=14, a2'=2", color="#00BFFF")
# Posterior #2: Diri(theta1, theta2; alpha1 = 14, alpha2 = 11)
plt.plot(theta_range, [Diri(1 - theta, theta, 14, 11) for theta in theta_range],
         label="Posterior a1'=14, a2'=11", color="#1E90FF")
# Posterior #3: Diri(theta1, theta2; alpha1 = 0.1, alpha2 = 0.2)
plt.plot(theta_range, [Diri(1 - theta, theta, 4.1, 1.2) for theta in theta_range],
         label="Posterior a1'=4.1, a2'=1.2", color="#104E8B")
plt.legend()
plt.xlabel("theta_2")
plt.ylabel("f(theta_2 -1, theta_2 ; alpha1, alpha2")
plt.show()

