


import numpy as np
# import multiprocessing as mp
from multiprocessing import Pool
import timeit
import time

def long_run(n):
    # s = ''
    # for i in range(int(n)):
    #     s += str(i)
    time.sleep(2)
    return n


def run_parallel():
    pool = Pool()
    results = [pool.apply(long_run, args=(x,)) for x in [5e5, 1e5, 7e5, 3e4, 2e4, 1e4, 2e5]]
    return results

def run_serial():
    results = [long_run(x) for x in [5e5, 1e5, 7e5, 3e4, 2e4, 1e4, 2e5]]
    return results

def simulate( times, K, eps, q_tot, c0, V, I_source):

    const = np.log(10)

    def dc_dt(c, t):
        c_eps = c[:, None] * eps  # hadamard product
        c_dot_eps = c_eps.sum(axis=0)

        q = c_eps * np.where(c_dot_eps <= 0.001, const - c_dot_eps * const * const / 2,
                             (1 - np.exp(-c_dot_eps * const)) / c_dot_eps) * I_source

        product = np.matmul(K, q.T[..., None]).squeeze()  # w x n x 1

        return q_tot / V * (product.sum(0) - (product[0] + product[-1]) / 2)

    return odeint(dc_dt, c0, times)

def log_likelihood( params, n_MCR_iter=5):

    # optimize spectra for curent C params by MCR-ALS style

    phi = self.Phi([params[0], params[1]], lambda_C=400)

    if any(phi < 0) or any(phi > 1):
        return -np.inf

    sigma = 0.01

    K = np.asarray([[-phi, self._0],
                    [+phi, self._0]])

    K = np.transpose(K, (2, 0, 1))
    C = np.zeros((self.times.shape[0], K.shape[0]))

    eps_est = self.eps_est.copy()

    for i in range(n_MCR_iter):
        # calc C
        C = self.simulate(self.times, K, eps_est, self.q_tot, self.c0, self.V, self.I_source)

        # calc ST by lstsq
        eps_est = lstsq(C, self.D)[0]

        # apply non-negative contraints on spectra
        eps_est *= (eps_est > 0)

    #         self.calls.append([params[0], self.eps_est])
    D_sim = C.dot(self.eps_est)
    residuals = self.D - D_sim

    # calculate the log of gaussian likelihood
    N = 1  # D.size
    #         LL = -0.5*N*np.log(2*np.pi*sigma**2) - (0.5/sigma**2) * (residuals**2).sum()

    LL = - (0.5 / sigma ** 2) * (residuals ** 2).sum()
    return LL









if __name__ == '__main__':

    # res = run_parallel()
    # res = run_serial()


    ret = timeit.timeit(lambda: run_serial(), number=2) / 2
    print(f"serial: {ret}")

    ret = timeit.timeit(lambda: run_parallel(), number=2) / 2
    print(f"parallel: {ret}")









