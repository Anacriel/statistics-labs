import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st
import math
from scipy.special import factorial

# Plotting parameters
sns.set(color_codes=True)
sns.set(font_scale=0.7)
axx = [(2, 4, 3), (2, 4, 4), (2, 4, 7), (2, 4, 8)]  # figures position

# Distributions display parameters
distrib_dict = {
    "Standard normal": {
        "color": "cadetblue",
        "interval": [-4, 4]},
    "Uniform": {
        "color": "seagreen",
        "interval": [-3, 3]},
    "Cauchy": {
        "color": "crimson",
        "interval": [-8, 8]},
    "Laplace": {
        "color": "darkslateblue",
        "interval": [-3, 3]},
    "Poisson": {
        "color": "mediumvioletred",
        "interval": [0, 15]}
}

n_vec = [20, 50, 100, 500]  # list of sample sizes


# Functions
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return st.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def get_uniform(x, param):
    y = x.copy()
    for i in range(len(x)):
        if abs(x[i]) < param:
            y[i] = 1 / (2 * param)
        else:
            y[i] = 0
    return y


def plot_result(distrib, color, x_a, x_b):
    fig = plt.figure(num=str(distrib) + "_th", figsize=(9, 5))
    x_dist = np.linspace(x_a, x_b, 3000)
    x = 0; y = 0

    if distrib == "Standard normal":
        y = st.norm.pdf(x_dist, 0, 1)
        x = get_truncated_normal(low=x_a, upp=x_b)
    elif distrib == "Uniform":
        y = get_uniform(x_dist, 3 ** 0.5)
        x = st.uniform(loc=-3 ** 0.5, scale=2 * (3 ** 0.5))
    elif distrib == "Cauchy":
        y = st.cauchy.pdf(x_dist, 0, 1)
        x = st.cauchy(loc=0, scale=1)
    elif distrib == "Laplace":
        y = st.laplace.pdf(x_dist, 0, (2 ** (-0.5)))
        x = st.laplace(loc=0, scale=(2 ** (-0.5)))
    elif distrib == "Poisson":
        y = np.exp(-7) * np.power(7, x_dist) / factorial(x_dist)
        x = st.poisson(mu=7)

    fig.add_subplot(2, 4, (1, 6))
    plt.plot(x_dist, y, label="Theoretical curve", color=color)
    plt.legend()
    plt.title(distrib + " distribution")

    i = 0
    for nrows, ncols, plot_number in axx:
        is_last = False

        s = fig.add_subplot(nrows, ncols, plot_number)
        in_bins = 1 + math.ceil(3.322 * math.log10(n_vec[i]))  # Sturges number

        sns.distplot(x.rvs(n_vec[i]), axlabel="n = " + str(n_vec[i]), bins=in_bins, color=color,  ax=s, kde=is_last,
                     hist=True, norm_hist=True)
        if i == len(n_vec) - 1:
            plt.plot(x_dist, y, label="Theoretical curve", color=color)
        i = i + 1
    plt.tight_layout()
    plt.show()


def main():
    for distrib, param in distrib_dict.items():
        plot_result(distrib=distrib, color=param["color"],
                    x_a=param["interval"][0], x_b=param["interval"][1])


if __name__ == "__main__":
    main()
