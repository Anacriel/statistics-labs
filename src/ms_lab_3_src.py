import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import numpy as np

# Plotting parameters
sns.set(color_codes=True)
sns.set(font_scale=0.7)

n_vec = [20, 100]  # list of sample sizes


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


# Functions
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


def count_outliers(data):
    data = np.sort(data)
    q1 = np.percentile(data, [25])
    q3 = np.percentile(data, [75])
    k = 1.5
    outliers = 0

    x_l = q1 - k * (q3 - q1)
    x_r = q3 + k * (q3 - q1)

    for elem in data:
        if (elem < x_l) or (elem > x_r):
            outliers = outliers + 1

    return outliers / len(data)  # percentage


def count_th_outliers(distrib, data):
    data = np.sort(data)
    q1 = distrib.ppf(0.25)
    q3 = distrib.ppf(0.75)
    k = 1.5
    outliers = 0

    x_l = q1 - k * (q3 - q1)
    x_r = q3 + k * (q3 - q1)

    for elem in data:
        if (elem < x_l) or (elem > x_r):
            outliers = outliers + 1

    return outliers / len(data)  # percentage


def plot_result(distrib, color):
    fig = plt.figure(num=str(distrib) + "_th", figsize=(6, 4))
    x = 0
    if distrib == "Standard normal":
        x = st.norm(loc=0., scale=1.)
    elif distrib == "Uniform":
        x = st.uniform(loc=-3 ** 0.5, scale=2 * (3 ** 0.5))
    elif distrib == "Cauchy":
        x = st.cauchy(loc=0, scale=1)
    elif distrib == "Laplace":
        x = st.laplace(loc=0, scale=(2 ** (-0.5)))
    elif distrib == "Poisson":
        x = st.poisson(mu=7)

    data_list = [x.rvs(n_vec[0]),
                 x.rvs(n_vec[1])]

    ax = sns.boxplot(data=data_list, color=color, orient="h")
    ax.get_yticklabels()
    ax.set_yticklabels(["n = 20", "n = 100"])

    outliers = [0, 0]
    th_outliers = [0, 0]

    for i in range(1000):
        data = [x.rvs(n_vec[0]),
                x.rvs(n_vec[1])]
        for k in range(len(data)):
            outliers[k] = outliers[k] + count_outliers(data[k])
            th_outliers[k] = th_outliers[k] + count_th_outliers(x, data[k])

    print(distrib + " distribution")
    print("n = 20")
    print("Outliers (sample): " + '%.4f' % (outliers[0] / 1000))
    print("Outliers (theoretical): " + '%.4f' % (th_outliers[0] / 1000))
    print("\nn = 100")
    print("Outliers (sample): " + '%.4f' % (outliers[1] / 1000))
    print("Outliers (theoretical): " + '%.4f' % (th_outliers[1] / 1000))

    plt.title(distrib + " distribution")
    plt.tight_layout()
    plt.show()


def main():
    for distrib, param in distrib_dict.items():
        plot_result(distrib=distrib, color=param["color"])


if __name__ == "__main__":
    main()


