import numpy as np
import scipy.stats as st
from collections import defaultdict


# Distributions display parameters
distributions = ("Standard normal", "Uniform", "Cauchy", "Laplace", "Poisson")
n_vec = [20, 50, 100]


# Functions
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return st.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def get_uniform(x, param):
    y = x.copy()
    for i in range(len(x)):
        if abs(x[i]) < param:
            y[i] = 1 / (2. * param)
        else:
            y[i] = 0.
    return y


def quartile_range(x, n):
    x_sorted = np.sort(x)
    q1 = np.median(x_sorted[:int(n / 2)])
    q3 = np.median(x_sorted[int(n / 2):])

    return (q3 + q1) / 2


def trunc_mean(x, n):
    x_sorted = np.sort(x)
    r = 0.1 * n

    return np.ma.sum(x_sorted[int(r):int(n - r)]) / (n - 2 * r)


def print_table(data, distrib, ind, n):
    moment_1 = list()
    moment_2 = list()

    for key, val in data[ind].items():
        moment_1.append(np.ma.sum(data[ind][key]) / 1000)
        moment_2.append(np.ma.sum((data[ind][key] - np.full(1000, moment_1[-1])) ** 2) / 1000)

    print(distrib)
    print("n = " + str(n))

    s_moment_1 = ' & '.join(['%.4f' % elem for elem in moment_1])
    s_moment_2 = ' & '.join(['%.4f' % elem for elem in moment_2])

    print("Mean      " + "Med    " + " Range    " + " IQR    " + " Truncated mean")
    print(s_moment_1)
    print(s_moment_2)


def get_characteristic(distrib, n):
    if distrib == "Standard normal":
        x_sample = st.norm(0, 1)
    elif distrib == "Uniform":
        x_sample = st.uniform(loc=-3 ** 0.5, scale=2 * (3 ** 0.5))
    elif distrib == "Cauchy":
        x_sample = st.cauchy(loc=0, scale=1)
    elif distrib == "Laplace":
        x_sample = st.laplace(loc=0, scale=(2 ** (-0.5)))
    elif distrib == "Poisson":
        x_sample = st.poisson(mu=7)
    else:
        x_sample = np.ndarray(shape=(1, n))

    result = defaultdict(list)

    for k in range(1000):
        x_n = x_sample.rvs(n)
        result["sample_mean"].append(np.mean(x_n))
        result["med"].append(np.median(x_n))
        result["range"].append((np.amax(x_n) + np.amin(x_n)) / 2)
        result["quart_range"].append(quartile_range(x_n, n))
        result["trunc_mean"].append(trunc_mean(x_n, n))

    return result


def calculate(distrib):
    distr_list = list()
    for i in range(len(n_vec)):
        distr_list.append(get_characteristic(distrib=distrib, n=n_vec[i]))
        print_table(data=distr_list, distrib=distrib, ind=i, n=n_vec[i])


if __name__ == "__main__":
    for distrib in distributions:
        calculate(distrib)





