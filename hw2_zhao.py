from matplotlib import pyplot as plt
import numpy as np


def simulation(p0m, p1m, p0M, p1M, s, mu_m, mu_M, t, current_e, r):
    if current_e == 0:
        w0 = 1
        w1 = 1 - s
        current_e = 1
    else:
        w0 = 1 - s
        w1 = 1
        current_e = 0
    for i in range(t):
        d = p0m * p1M - p0M * p1m
        w_bar = w1 * (p1m + p1M) + w0 * (p0m + p0M) + (w0 + w0 + w1 + w1) * r * d
        n_p0m = (w1 * (mu_m) * p1m + w0 * (1 - mu_m) * p0m + w0 * r * d) / w_bar
        n_p1m = (w1 * (1 - mu_m) * p1m + w0 * (mu_m) * p0m + w1 * r * d) / w_bar
        n_p0M = (w1 * (mu_M) * p1M + w0 * (1 - mu_M) * p0M + w0 * r * d) / w_bar
        n_p1M = (w1 * (1 - mu_M) * p1M + w0 * (mu_M) * p0M + w1 * r * d) / w_bar
        p0m = n_p0m
        p1m = n_p1m
        p0M = n_p0M
        p1M = n_p1M
    return p0m, p1m, p0M, p1M, current_e


if __name__ == '__main__':
    N = 20
    generations = 100000
    simulations = 400
    burn_in = 1000
    switching_times = 150
    # s0 == s1
    s = 0.01

    # recombination rate
    r = 0.01

    x = []
    y = []

    for var in range(1, simulations + 1, 5):
        # gamma dist parameters, mean = 20, and variance from 1 - 400
        a = 400 / var
        b = 20 / a

        current_e = 0

        # random choose init p0 start
        p0m = .2
        p1m = .8
        p0M = 0
        p1M = 0

        # random choose init mu_m
        mu_m = np.random.random_sample() / 10  # init mu_m between 0-0.1
        mu_M = np.random.random_sample() / 10  # init mu_M between 0-0.1

        for j in range(switching_times):
            # print(j)
            # start simulation to find switching rate
            i = 0

            while i < burn_in:
                # sample from gamma the time before changing the env
                t = int(np.floor(np.random.gamma(a, b)))
                p0m, p1m, p0M, p1M, current_e = simulation(p0m, p1m, p0M, p1M, s, 0, 0, t, current_e, 0)
                i = i + t

            p0M = 0.0001 * p0m
            p1M = 0.0001 * p1m
            p0m = 0.9999 * p0m
            p1m = 0.9999 * p1m

            while i < generations and (p1M + p0M != 1 or p1M + p0M != 0):
                t = int(np.floor(np.random.gamma(a, b)))
                p0m, p1m, p0M, p1M, current_e = simulation(p0m, p1m, p0M, p1M, s, mu_m, mu_M, t, current_e, r)
                i = i + t

            if (p0M + p1M > 0.0001):
                # invasion successful
                mu_m = mu_M
            mu_M = np.random.exponential(1) * mu_m

        print(var, np.log10(mu_m))
        x.append(var)
        y.append(np.log10(mu_m))

    f, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y)
    plt.show()
