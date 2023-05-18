import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define distributions
def gamma(x):
    return np.factorial(x - 1)


# Dirichlet/ Beta prior with alpha1 and alpha2 data
def Diri(theta1, theta2, alpha1, alpha2):
    return gamma(alpha1 + alpha2) / (gamma(alpha1) * gamma(alpha2)) * \
           theta1 ** (alpha1 - 1) * theta2 ** (alpha2 - 1)


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
