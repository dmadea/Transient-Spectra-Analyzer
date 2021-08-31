import numpy as np
import matplotlib.pyplot as plt
from numpy import gradient
from scipy.integrate import cumtrapz
from numpy.linalg import lstsq


# from https://math.stackexchange.com/questions/1428566/fit-sum-of-exponentials/3808325#3808325

x = np.linspace(0, 10, 500)
y = 1.5 * np.exp(0.5 * x) - 5 * np.exp(0.35 * x)

ynoise = np.random.normal(y, 0.1)

xR, yR = x[::100], ynoise[::100]


iy1 = cumtrapz(yR, xR, initial=0)
iy2 = cumtrapz(iy1, xR, initial=0)


Y = np.asarray([iy1, iy2, xR, np.ones_like(xR)]).T

A = lstsq(Y, yR, rcond=None)[0]


Ahat = np.diag(np.ones(A.shape[0] // 2 - 1), -1)
Ahat[0] = A[:A.shape[0] // 2]

# print(Ahat)

lambdas = np.linalg.eigvals(Ahat)
if any(np.iscomplex(lambdas)):
    lambdas = lambdas.real
print(lambdas)

X = np.exp(lambdas[None, :] * xR[:, None])

multipliers = lstsq(X, yR, rcond=None)[0]
print(multipliers)

best_fit = multipliers[None, :] * np.exp(lambdas[None, :] * x[:, None])
best_fit = best_fit.sum(axis=-1, keepdims=False)

plt.scatter(xR, yR)
plt.plot(x, y, label='original')
plt.plot(x, best_fit, lw=2, ls='--', color='k')
plt.legend()
plt.show()