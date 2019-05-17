import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as st
from collections import defaultdict
from scipy.stats import multivariate_normal


# Plotting parameters
sns.set_style("darkgrid")
sns.set(color_codes=True)
sns.set(font_scale=0.7)

# Data parameters
n_vec = [20, 60, 100]
cov_vec = [[(1, 0), (0, 1)], [(1, 0.5), (0.5, 1)], [(1, 0.9), (0.9, 1)]]
coefficients = ("pearson", "spearman", "quad")


# Functions
def show_coeffs(data, head):
    mean = defaultdict(float)
    mean_sq = defaultdict(float)
    disp_sum = defaultdict(float)

    for coeff in coefficients:
        mean[coeff] = sum(data[coeff]) / 1000
        mean_sq[coeff] = sum(map(lambda el: el * el, data[coeff])) / 1000
        disp_sum[coeff] = mean_sq[coeff] - mean[coeff] ** 2

    print(head)
    print("Mean: " + ' & '.join(['%.4f' % val for key, val in mean.items()]))
    print("Mean squares: " + ' & '.join(['%.4f' % val for key, val in mean_sq.items()]))
    print("Dispersion: " + ' & '.join(['%.4f' % val for key, val in disp_sum.items()]) + "\n")


def count_distrib_sum(n):
    corr_sum = defaultdict(list)

    for f in range(1000):
        dist_1 = multivariate_normal(mean=[0, 0], cov=[(1, 0.9), (0.9, 1)])
        dist_2 = multivariate_normal(mean=[0, 0], cov=[(10, -9), (-9, 10)])

        dist_sum = 0.9 * dist_1.rvs(n) + 0.1 * dist_2.rvs(n)
        dist_sum = dist_sum.transpose()

        _v, _p = st.pearsonr(dist_sum[0], dist_sum[1])
        corr_sum["pearson"].append(_v)
        _v, _p = st.spearmanr(dist_sum[0], dist_sum[1])
        corr_sum["spearman"].append(_v)
        corr_sum["quad"].append(quar_coef(dist_sum[0], dist_sum[1], n))

    return corr_sum


def quar_coef(x, y, n):
    q_1 = 0; q_2 = 0; q_3 = 0; q_4 = 0

    for i in range(n):
        if x[i] > 0 and y[i] > 0:
            q_1 += 1
        elif x[i] < 0 and y[i] > 0:
            q_2 += 1
        elif x[i] < 0 and y[i] < 0:
            q_3 += 1
        elif x[i] > 0 and y[i] < 0:
            q_4 += 1

    return (q_1 + q_3 - (q_2 + q_4)) / n


def plot_ellipse(n, cov):
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    x, y = np.random.multivariate_normal([0, 0], cov, n).T

    plt.figure(num="n" + str(n) + "_" + "ro" + "0_" + str(int(10 * cov[0][1])), figsize=(5, 5))

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    ax = plt.gca()
    plt.scatter(x, y, alpha=0.7, color="salmon")
    ell = Ellipse(xy=(0, 0),
                  width=lambda_[0] * 3 * 2, height=lambda_[1] * 3 * 2,
                  angle=np.rad2deg(np.arccos(v[0, 0])), color="mediumturquoise")
    ell.set_facecolor('none')
    ax.add_artist(ell)

    plt.xlabel("x")
    plt.xlabel("y")

    plt.title("99% ellipse\nn = " + str(n) + ", correlation = " + str(cov[0][1]))


def main():
    for n in n_vec:
        for cov in cov_vec:
            plot_ellipse(n, cov)

            correlations = defaultdict(list)
            for k in range(1000):
                x, y = np.random.multivariate_normal([0, 0], cov, n).T
                val, p_val = st.pearsonr(x, y)
                correlations["pearson"].append(val)
                val, p_val = st.spearmanr(x, y)
                correlations["spearman"].append(val)
                correlations["quad"].append(quar_coef(x, y, n))

            show_coeffs(correlations, head="n = " + str(n) + ", ro = " + str(str(cov[0][1])))
            plt.tight_layout()
            plt.show()

        correlations_sum = count_distrib_sum(n)
        show_coeffs(correlations_sum, head="Sum of distributions" + "\n" + "n = " + str(n))


if __name__ == '__main__':
    main()
