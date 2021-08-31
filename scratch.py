import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from numpy.linalg import lstsq

x = np.linspace(0, 10, 500)

m = np.asarray([1.5, -5, 50])
l = np.asarray([0.5, 0.35, 0])

y = (m[None, :] * np.exp(l[None, :] * x[:, None])).sum(axis=-1)

ynoise = np.random.normal(y, 0)

xR, yR = x[::20], ynoise[::20]

# from https://math.stackexchange.com/questions/1428566/fit-sum-of-exponentials/3808325#3808325
def fit_sum_exp(x, y, n=2, fit_intercept=False):
    """Fits the data with the sum of exponential function and returns

    if fit_intercept is True, last multiplier will be the intercept, also the 0 will be added
    at the end of lambda vector"""

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] >= 2 * n

    Y_size = 2 * n + 1 if fit_intercept else 2 * n
    Y = np.empty((x.shape[0], Y_size))

    Y[:, 0] = cumtrapz(y, x, initial=0)
    for i in range(1, n):
        Y[:, i] = cumtrapz(Y[:, i - 1], x, initial=0)

    Y[:, -1] = 1
    for i in reversed(range(n, Y_size - 1)):
        Y[:, i] = Y[:, i + 1] * x

    A = lstsq(Y, y, rcond=None)[0]
    Ahat = np.diag(np.ones(n - 1), -1)
    Ahat[0] = A[:n]

    lambdas = np.linalg.eigvals(Ahat)
    # remove complex values
    if any(np.iscomplex(lambdas)):
        lambdas = lambdas.real

    X = np.exp(lambdas[None, :] * x[:, None])
    if fit_intercept:
        X = np.hstack((X, np.ones_like(x)[:, None]))
        lambdas = np.insert(lambdas, n, 0)
    multipliers = lstsq(X, y, rcond=None)[0]

    return multipliers, lambdas


multipliers, lambdas = fit_sum_exp(xR, yR, 2, True)
print(multipliers, lambdas)

best_fit = multipliers[None, :] * np.exp(lambdas[None, :] * x[:, None])
best_fit = best_fit.sum(axis=-1, keepdims=False)

plt.scatter(xR, yR)
plt.plot(x, y, label='original')
plt.plot(x, best_fit, lw=2, ls='--', color='k')
plt.legend()
plt.show()

