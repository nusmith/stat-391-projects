import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Logistic regression for classification
# P(X | Y = 0) ~ N(0, s^2) and P(X | Y = 1) ~ N(1, s^2)
# Normal distribution with mu u and sd s.
def N(x, u, s):
    return 1 / (s*np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x-u)/s) ** 2)

# P(Y=1|X=x) is the likelihood function for the parameter Y = 1 given data x
x_series = np.arange(-5, 5, 0.1)
plt.plot(x_series, [N(x, u=1, s=0.5) for x in x_series], )
plt.xlim((-5, 5))
plt.vlines(x=[0,1], ymin=0, ymax=1, label="Class means u0, u1", colors="red")
# Area for decision boundary
x_series = np.arange(-2, 2, 0.1).round(1)
y1 = pd.DataFrame({'x': x_series, 'py': [N(x, u=1, s=0.5) for x in x_series]})
y0 = pd.DataFrame({'x': x_series, 'py': [N(x, u=0, s=0.5) for x in x_series]})
condition = np.round([0.5 - 0.5**2*y for y in np.log(np.divide(y1, y0)).py], 1)
x_range_condition = [x for x in x_series if x in condition]
print(x_range_condition)
plt.fill_between(y1=[N(x, u=1, s=0.5) for x in x_range_condition],
                 x=x_range_condition, color='green', alpha=.2, label="Boundary condition region")
plt.vlines(x=[np.min(x_range_condition), np.max(x_range_condition)], ymin=0, ymax=1,
           label="Boundary condition", colors="green", alpha=.2)
plt.legend()
plt.title("P(Y=1|X=x)")
plt.xlabel("x")
plt.ylabel("L(Y | x)")
plt.show()
plt.plot()
