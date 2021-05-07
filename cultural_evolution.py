#!/usr/bin/env python3

import random
import time
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from sys import argv
from typing import Tuple, List

import numpy as np
from scipy.special import comb


# hyper-parameters
# s1 = 1
# s2 = 1
# p_o = 0.1
# mu = 1e-4
# c = 10
# n = 5
# learning_flag = "biased_R"


# heterozygosity parameters
# delta_c = 0.01


def prob_helper(m, n, summand):
    m = int(m + 0.5)
    assert n >= m
    return sum(comb(n, mi, exact=False) * summand ** mi * (1 - summand) ** (n - mi) for mi in range(m, n + 1))


def oblique_transmission(x, xv, p_o, n=1e3, learning_flag="random"):
    x1, x2, x3, x4 = x
    x1v, x2v, x3v, x4v = xv

    if learning_flag == "random":
        favor_1_2 = x1 + x2
        favor_3_4 = x3 + x4
    elif learning_flag == "conformist":
        favor_1_2 = prob_helper(n / 2 + 1, n, x1 + x2)
        favor_3_4 = prob_helper(n / 2 + 1, n, x3 + x4)
    elif learning_flag == "biased_R":
        favor_1_2 = prob_helper(2, n, x1 + x2)
        favor_3_4 = prob_helper(n, n, x3 + x4)
    elif learning_flag == "biased_r":
        favor_1_2 = prob_helper(n, n, x1 + x2)
        favor_3_4 = prob_helper(2, n, x3 + x4)
    else:
        raise NotImplementedError

    x1o = (1 - p_o) * x1v + p_o * (x1v + x3v) * favor_1_2
    x2o = (1 - p_o) * x2v + p_o * (x2v + x4v) * favor_1_2
    x3o = (1 - p_o) * x3v + p_o * (x1v + x3v) * favor_3_4
    x4o = (1 - p_o) * x4v + p_o * (x2v + x4v) * favor_3_4

    xo = np.array([x1o, x2o, x3o, x4o])
    return xo


def learning(x: Tuple[float, float, float, float], p_o, n=1e3, learning_flag="random"):
    x1, x2, x3, x4 = x

    # vertical transmission
    summand = (x1 * x4 + x2 * x3) / 2.0
    x1v = x1 ** 2 + (x1 * x2 + x1 * x3 + summand)
    x2v = x2 ** 2 + (x1 * x2 + x2 * x4 + summand)
    x3v = x3 ** 2 + (x1 * x3 + x3 * x4 + summand)
    x4v = x4 ** 2 + (x2 * x4 + x3 * x4 + summand)
    xv = np.array([x1v, x2v, x3v, x4v])

    return oblique_transmission(x, xv, p_o, n, learning_flag)


def innovation(xo, mu):
    mu_M, mu_m = mu
    xo_1, xo_2, xo_3, xo_4 = xo
    xi_1 = mu_M * xo_3 + (1 - mu_M) * xo_1
    xi_2 = mu_m * xo_4 + (1 - mu_m) * xo_2
    xi_3 = mu_M * xo_1 + (1 - mu_M) * xo_3
    xi_4 = mu_m * xo_2 + (1 - mu_m) * xo_4

    return np.array([xi_1, xi_2, xi_3, xi_4])


def selection(xi, env, fitness):
    xi_1, xi_2, xi_3, xi_4 = xi
    bar_omega = sum(np.array([xi_1 + xi_2, xi_3 + xi_4]) * fitness[env])
    return xi * np.repeat(fitness[env], 2) / bar_omega


def oblique_transmission_heterozygosity(x, xv, p_o, n=1e3, learning_flag="random"):
    x1, x2, x3 = x
    x1v, x2v, x3v = xv

    if learning_flag == "random":
        x1o = (1 - p_o) * x1v + p_o * (x1v + x2v) * (x1 + x2 / 2)
        x2o = (1 - p_o) * x2v + p_o * x1v * (x3 + x2 / 2) + p_o * x3v * (x1 + x2 / 2)
        x3o = (1 - p_o) * x3v + p_o * (x2v + x3v) * (x2 / 2 + x3)
    elif learning_flag == "biased_R":
        x1o = (1 - p_o) * x1v + p_o * (x1v + x2v / 2) * prob_helper(2, n, x1 + x2)
        x2o = (1 - p_o) * x2v + p_o * (x2v / 2 + x3v) * prob_helper(2, n, x1 + x2)
        x3o = (1 - p_o) * x3v + p_o * x3v * (1 - prob_helper(2, n, x1 + x2))
    elif learning_flag == "biased_r":
        x1o = (1 - p_o) * x1v + p_o * x1v * (1 - prob_helper(2, n, x2 + x3))
        x2o = (1 - p_o) * x2v + p_o * (x1v + x2v / 2) * prob_helper(2, n, x2 + x3)
        x3o = (1 - p_o) * x3v + p_o * (x2v / 2 + x3v) * prob_helper(2, n, x2 + x3)
    else:
        raise NotImplementedError

    xo = np.array([x1o, x2o, x3o])
    return xo


def learning_heterozygosity(x, p_o, n=1e3, learning_flag="random"):
    x1, x2, x3 = x

    # vertical transmission
    x1v = x1 ** 2 + x1 * x2 + x2 ** 2 / 4
    x2v = x2 ** 2 / 2 + 2 * x1 * x3 + x1 * x2 + x2 * x3
    x3v = x3 ** 2 + x2 * x3 + x2 ** 2 / 4
    xv = np.array([x1v, x2v, x3v])

    return oblique_transmission(x, xv, p_o, n, learning_flag)


def innovation_heterozygosity(mu, xo):
    xo1, xo2, xo3 = xo
    xi1 = (1 - mu) * xo1 + mu * xo2 / 2
    xi2 = (1 - mu) * xo2 + mu * (xo1 + xo3)
    xi3 = (1 - mu) * xo3 + mu * xo2 / 2

    return np.array([xi1, xi2, xi3])


def selection_heterozygosity(xi, env, s1, s2, delta_c):
    xi_1, xi_2, xi_3, xi_4 = xi
    if env == 0:
        fitness_weight = np.array([1, 1 + s1, 1 + s1])
        bar_omega = sum(np.array([xi_1, xi_2 * (1 - delta_c), xi_3]) * fitness_weight)
        x_prime_1 = xi_1 / bar_omega
        x_prime_2 = xi_2 * (1 - delta_c) * (1 + s1) / bar_omega
        x_prime_3 = xi_3 * (1 + s1) / bar_omega
    else:
        fitness_weight = np.array([1 + s2, 1 + s2, 1])
        bar_omega = sum(np.array([xi_1, xi_2 * (1 - delta_c), xi_3]) * fitness_weight)
        x_prime_1 = xi_1 * (1 + s2) / bar_omega
        x_prime_2 = xi_2 * (1 - delta_c) * (1 + s2) / bar_omega
        x_prime_3 = xi_3 / bar_omega

    return np.array([x_prime_1, x_prime_2, x_prime_3])


def get_rand_rate_innovation():
    return random.uniform(0, 1)


def stable(generation) -> bool:
    """
    check for stable equilibrium
    :return:
    """
    return generation > 1e3


def evolve(x, mu, p_o, n, learning_flag, env, fitness):
    xo = learning(x, p_o, n, learning_flag)
    # print("learning", xo)
    xi = innovation(xo, mu)
    # print("innovation", xi)
    x = selection(xi, env, fitness)
    # print("selection", x)
    return x


def evolve_heterozygosity(x, mu, p_o, env, s1, s2, delta_c) -> np.array:
    xo = learning(x, p_o)
    xi = innovation(xo, mu)
    return selection_heterozygosity(xi, env, s1, s2, delta_c)


def time_step(generation, env, c):
    generation += 1
    if generation % c == 0:
        env = ~env
    return generation, env


def simulation(env, p_o, c, n, learning_flag, fitness):
    start_time = time.time()
    post_stability_generations = 1000
    stable_tenure = 300

    current_rate_tenure = 0
    mu_m, mu_M = np.random.uniform(0.1, size=2)
    intro_rate = 1e-4
    iterations = 0
    while current_rate_tenure < stable_tenure:
        generation = 0
        iterations += 1

        x_1 = get_rand_rate_innovation()
        x_1 = max(x_1, 1-x_1)
        x_3 = 1 - x_1

        x = np.array([x_1, 0, x_3, 0])
        # burn in period
        # let population stabilize first
        # print("burn-in started!!!")
        while not stable(generation):
            x = evolve(x, mu=(mu_M, mu_m), p_o=p_o, n=n, learning_flag=learning_flag, env=env, fitness=fitness)
            generation, env = time_step(generation, env, c)  # maybe update env
            # print(generation)
        # print("burn-in ended!!!")

        # introduce m
        # print("introduce m!!!")
        x_1, x_3 = x[0], x[2]
        x = np.array([x_1, x_1, x_3, x_3]) * np.tile([1 - intro_rate, intro_rate], 2)

        # run for 1k extra generations
        for _ in range(post_stability_generations):
            x = evolve(x, mu=(mu_M, mu_m), p_o=p_o, n=n, learning_flag=learning_flag, env=env, fitness=fitness)
            generation, env = time_step(generation, env, c)  # maybe update env

        # check if an invasion occurred
        # f(m) = f(Rm) + f(rm) = x2 + x4
        m = x[1] + x[3]
        if m < intro_rate:
            # invasion did not occur
            current_rate_tenure += 1
        else:
            # invasion occurred
            # set new resident value
            mu_M = mu_m
            current_rate_tenure = 0

        mu_m = mu_M * np.random.exponential(1.)
        # if current_rate_tenure > 200:
        print(f"current_rate_tenure {current_rate_tenure}")
        if iterations > 3000:
            print("BREAKING")
            break

    end_time = time.time()
    print(f"it took {end_time - start_time} seconds to finish")
    return str(mu_M)  # "\t".join(map(str, mu_M.tolist()))


class Env(int):
    def __init__(self, val):
        self.environment = val

    def __invert__(self):
        return -(self.environment - 1)


# seed = 1966
# random.seed(seed), np.random.seed(seed)

# id = random.randrange(10000)
# with open(f"results_{id}.out", "w") as out:
#     out.writeline(" ".join(argv))
#     out.writelines(results)

def run_simulation(line: str) -> np.array:
    """
    run a simulation
    :param line: the parameter string from the input file
    :return:
    """
    # random.seed(1996), np.random.seed(1996)
    s1, s2, p_o, mu, c, n, delta_c = list(map(float, line.split()[:-1]))
    c, n = int(c), int(n)
    learning_flag = line.split()[-1]  # "biased_R"
    fitness = np.array([[1, 1 + s1], [1 + s2, 1]])

    environment = Env(random.choice([0, 1]))
    num_simulations = 2
    results: List[str] = [simulation(environment, p_o, c, n, learning_flag, fitness) for _ in range(num_simulations)]
    id = random.randrange(10000)
    with open(f"results_{id}.out", "w") as out:
        out.writelines([line, "\n", "\n".join(results), "\n"])
    return line, results


# run_simulation("1 1 0.1 1e-4 10 5 0.01 biased_R")

if __name__ == '__main__':
    assert len(argv) == 2, "need config file name"

    with open(argv[1]) as config_file:
        configs = [l.strip() for l in config_file.readlines()]

    with Pool(cpu_count() - 1) as pool:
        results = dict(pool.map(run_simulation, configs))

    with open("result.out", "a") as out:
        for line, outputs in results.items():
            out.writelines([line, "\n", "\n".join(outputs), "\n\n"])
