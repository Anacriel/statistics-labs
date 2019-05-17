import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import numpy as np
from scipy.special import factorial


# Plotting parameters
sns.set(color_codes=True)
sns.set(font_scale=0.7)


# Distributions display parameters
distrib_dict = {
    "Standard normal": {
        "color": "cadetblue"},
    "Uniform": {
        "color": "seagreen"},
    "Cauchy": {
        "color": "crimson"},
    "Laplace": {
        "color": "darkslateblue"},
    "Poisson": {
        "color": "mediumvioletred"}
}

n_vec = [20, 60, 100]  # list of sample sizes


# Functions
def get_uniform(x, param):
    y = x.copy()
    for i in range(len(x)):
        if abs(x[i]) < param:
            y[i] = 1 / (2. * param)
        else:
            y[i] = 0.
    return y


def ecdf(x):
    n = len(x)

    if sorted(x) is not x:
        x = np.sort(x)

    y = np.arange(1, n + 1) / n

    return y


def plot_result(color, distrib, x_a=-4, x_b=4):
    x = 0; y = 0
    x_dist = np.linspace(x_a, x_b, 3000)
    if distrib == "Standard normal":
        y = st.norm.cdf(x_dist, 0, 1)
        y_p = st.norm.pdf(x_dist, 0, 1)
        x = st.norm(loc=0., scale=1.)
    elif distrib == "Uniform":
        y_p = st.uniform.pdf(x_dist, -3 ** 0.5, 2 * (3 ** 0.5))
        y = st.uniform.cdf(x_dist, -3 ** 0.5, 2 * (3 ** 0.5))
        x = st.uniform(loc=-3 ** 0.5, scale=2 * (3 ** 0.5))
    elif distrib == "Cauchy":
        y = st.cauchy.cdf(x_dist, 0, 1)
        y_p = st.cauchy.pdf(x_dist, 0, 1)
        x = st.cauchy(loc=0, scale=1)
    elif distrib == "Laplace":
        y = st.laplace.cdf(x_dist, 0, (2 ** (-0.5)))
        y_p = st.laplace.pdf(x_dist, 0, (2 ** (-0.5)))
        x = st.laplace(loc=0, scale=(2 ** (-0.5)))
    elif distrib == "Poisson":
        y = st.poisson.cdf(x_dist, mu=2)
        y_p = np.exp(-2) * np.power(2, x_dist) / factorial(x_dist)
        x = st.poisson(mu=2)

    for i in n_vec:
        sample_x = np.sort(x.rvs(i))

        cdf = ecdf(sample_x)

        plt.step(sample_x, cdf, color="darkslategrey", label="Empirical distribution function")
        plt.plot(x_dist, y, color=color)

        plt.legend()
        plt.xlim(x_a, x_b)
        plt.tight_layout()
        plt.show()

    for i in n_vec:
        sample_x = np.sort(x.rvs(i))

        fig = plt.figure(num="ker_" + str(distrib) + "_" + str(i), figsize=(11, 4))
        ind = 1
        for h in [1, 3, 5]:
            ax = fig.add_subplot(1, 3, ind)
            sns.kdeplot(sample_x, color="darkslategrey", bw=h, ax=ax,
                        label=("Kernel density estimation" if h == 1 else ""))
            if h == 5:
                ax.legend(["Kernel density estimation"], loc="upper center", bbox_to_anchor=(0.5, -0.25))

            plt.plot(x_dist, y_p, color=color)

            plt.xlim(x_a, x_b)
            plt.title("h = " + str(h))
            ind = ind + 1

        plt.tight_layout()
        plt.show()


def main():
    for distrib, param in distrib_dict.items():
        plot_result(distrib=distrib, color=param["color"])


if __name__ == "__main__":
    main()


