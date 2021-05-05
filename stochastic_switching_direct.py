import random
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from typing import Tuple

import numpy as np

# variable
environments = (env_0 := 0, env_1 := 1)
major_alleles = (major_0 := 0, major_1 := 1)
modifier_alleles = (m := 0, M := 1)


def fitness(individual: int, env: int) -> float:
    return 1 if individual == env else 1 - pheno_malfitness[individual]


def get_genotype_freqencies(pop: np.array) -> Tuple[float, float]:
    f_1 = pop["major"].mean()
    f_0 = 1 - f_1
    return f_0, f_1


def new_genotype_frequencies(pop: np.array, env: int) -> np.array:
    w_0x, w_1x = fitness(major_0, env), fitness(major_1, env)
    f_0, f_1 = get_genotype_freqencies(pop)
    mean_fitness = f_0 * w_0x + f_1 * w_1x
    return np.array([w_0x, w_1x]) / mean_fitness


def mutate(pop: np.array) -> np.array:
    # determine who will mutate @ major
    pop_mutation_rates = np.where(pop["modifier"] == M, mu_M, mu_m)
    mutants = np.random.uniform(size=population_size) <= pop_mutation_rates

    # determine who will mutate @ modifier
    pop_mutation_rates = np.where(pop["modifier"] == m, mut_m_to_M, mut_M_to_m)
    mod_mutants = np.random.uniform(size=population_size) <= pop_mutation_rates

    # create mutant-only pools to pick from (mutate everyone)
    mutant_pool = (pop["major"] + 1) % len(major_alleles)
    mod_mutant_pool = (pop["modifier"] + 1) % len(modifier_alleles)

    # mutate the population
    mutant_pop = pop.copy()
    mutant_pop["major"] = np.where(mutants, mutant_pool, pop["major"])
    mutant_pop["modifier"] = np.where(mod_mutants, mod_mutant_pool, pop["modifier"])

    return mutant_pop


def select_reproduce(pop: np.array, env: int) -> np.array:
    fitnesses = get_fitnesses(pop["major"], env)
    probabilities = fitnesses / fitnesses.sum()

    # sample the population according to their fitnesses
    new_pop = np.random.choice(pop, size=population_size, p=probabilities)
    return new_pop


def get_environment_change_schedule(total_time: int, average: int = 10, gamma=True, gamma_scale=1.0) -> np.array:
    if gamma:
        schedule = np.array([])
        while schedule.sum() < total_time:
            schedule = np.append(schedule, round(np.random.gamma(average, scale=gamma_scale)))
        return schedule.cumsum()[:-1]  # ignore last entry, it will be over the total time
    else:
        return np.arange(total_time, step=average)


def create_population(major_distribution=(0.5, 0.5)):
    return np.array(
        list(zip(
            np.random.choice(major_alleles, population_size, p=major_distribution),  # major
            np.random.choice(modifier_alleles, population_size, p=[1 - modifier_freq, modifier_freq]))  # modifier
        ),
        dtype=[('major', 'int8'), ('modifier', 'int8')]
    )


def simulate(seed=1966, use_gamma_intervals: bool = True):
    random.seed(seed)
    np.random.seed(seed)

    environment = random.choice(environments)
    population = create_population()
    init = f"init:\t" \
           f"major: {population['major'].mean()}\t" \
           f"modifier: {population['modifier'].mean()}"
    env_change_schedule = set(get_environment_change_schedule(total_time=time_steps,
                                                              average=env_change_interval,
                                                              gamma=use_gamma_intervals))
    for step in range(time_steps):
        # environment change check
        if step in env_change_schedule:
            environment = (environment + 1) % len(environments)

        # clonal reproduction & selection & mutation
        population = mutate(select_reproduce(population, environment))

        proportion_M = population["modifier"].mean()
        if np.isclose(proportion_M, 1) or np.isclose(proportion_M, 0):
            break

    final = f"end:\t" \
            f"major: {population['major'].mean()}\t" \
            f"modifier: {population['modifier'].mean()}"
    return init, final


get_fitnesses = np.vectorize(fitness, otypes=[float])

# primary parameters
runs = cpu_count() - 1

population_size = int(10e3)
time_steps = int(100e3)
modifier_freq = 1e-4
env_change_interval = 100

pheno_malfitness = np.array([  # fitness outside of home env
    s_0 := 0.5,
    s_1 := 0.5
])

major_mutation_rates = np.array([
    mu_m := 1e-3,
    mu_M := 1e-1
])

modifier_mutation_rates = np.array([
    mut_m_to_M := modifier_freq,
    mut_M_to_m := modifier_freq
])


def get_mutation_rates(size):
    rates = list(np.random.uniform(0, 0.1, size=2))
    for _ in range(size - 1):
        rates.append(rates[-1] * np.random.gamma(1))

    pairs = [tuple(rates[i:i + 2]) for i in range(size)]
    return pairs


if __name__ == "__main__":

    rand_seed = 1966
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    seeds = np.random.randint(rand_seed, size=runs)
    mutation_pairs = get_mutation_rates(size=runs)

    with Pool(cpu_count() - 1) as pool:
        results = pool.map(simulate, seeds)
        for first, second in results:
            print(f"{first}\n{second}", end="\n\n")
