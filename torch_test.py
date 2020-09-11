
import torch
import numpy as np
import timeit
import time
from scipy import linalg
from scipy.linalg import lstsq as _lstsq

posv = linalg.get_lapack_funcs(('posv'))


def direct_solve(a, b):
    c, x, info = posv(a, b, lower=False,
                      overwrite_a=True,
                      overwrite_b=False)
    return x


def lstsq(A, b, use_scipy=False):
    if use_scipy:
        return _lstsq(A, b)[0]
    else:
        if A.ndim == 3:
            ret_mat = np.empty((A.shape[-1], A.shape[0]))
            for i in range(A.shape[0]):
                ret_mat[:, i] = direct_solve(A[i].T.dot(A[i]), A[i].T.dot(b[:, i]))

            return ret_mat
        return direct_solve(A.T.dot(A), A.T.dot(b))


def b_lstsq(A, B):
    """
    Batched linear least-squares by numpy with direct solve method.
    Minimizes sum ||A_i x - B_i||_2^2 for x, batchwise

    Parameters
    ----------
        A : shape(L, M, N)
        B : shape(M, L)

    Returns
    -------
        tuple of (coefficients, fit, residuals)
    """

    AT = np.transpose(A, (0, 2, 1))  # transpose of A
    ATA = np.matmul(AT, A)

    ATB = np.matmul(AT, B.T[..., None])

    X = np.linalg.solve(ATA, ATB)

    fit = np.matmul(A, X).squeeze().T
    res = fit - B

    return X.squeeze().T, fit, res


# def lstsq_torch(A, b):
#
#     A_t = torch.from_numpy(A)
#     b_t = torch.from_numpy(b)
#
#     ATA = A_t.t() @ A_t   #torch.bmm(A_t.t(), A_t)
#     ATb = A_t.t() @ b_t
#
#     print(ATA)
#     print(ATb)
#
#
#     X, LU = torch.solve(ATA, ATb)
#     #
#     # return X




if __name__ == '__main__':

    B = np.random.random((300, 400))
    A = np.random.random((400, 300, 3))

    # x_fast = lstsq(A, B, False)
    x_numpy, fit, res = b_lstsq(A, B)

    # print(x_numpy.shape, fit.shape)
    # x_gesv = lstsq(A, B, True)
    # x_numpy = lstsq_numpy(A, B)


    # print(np.allclose(x_fast,x_numpy))
    # print(np.allclose(x_fast,x_gesv))
    # print(np.allclose(x_numpy,x_gesv))


    # x_torch = lstsq_torch(A, B)
    #
    # n = 1000
    # ret = timeit.timeit(lambda: b_lstsq(A, B), number=n)
    # print(ret / n)
    #
    # ret = timeit.timeit(lambda: lstsq(A, B, True), number=n)
    # print(ret)
    #
    # ret = timeit.timeit(lambda: lstsq_numpy(A, B), number=n)
    # print(ret)
    #
    # #
    # print(np.allclose(x_fast, x_gesv, x_numpy))
    # print(x_fast)
    # print(x_gesv)


    #
    # ret = timeit.timeit(lambda: run_t(m), number=100000)
    # print(ret / 100000)

    # ret = timeit.timeit(lambda: run_np(), number=100)
    # print(ret)


