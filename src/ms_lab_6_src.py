import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st


n_size = 20
sns.set_style("darkgrid")


def ethalon_function(x):
    return 2. + 2. * x + st.norm(loc=0., scale=1.).rvs(n_size)


def estimate_least_squares(x, y):
    n = np.size(x)

    m_x, m_y = np.mean(x), np.mean(y)

    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    theta_1 = SS_xy / SS_xx
    theta_0 = m_y - theta_1 * m_x

    return theta_0, theta_1


def estimate_least_deviations(x, y):
    n = np.size(x)

    med_x, med_y = np.median(x), np.median(y)

    iqr_x = np.percentile(x, [75]) - np.percentile(x, [25])
    iqr_y = np.percentile(y, [75]) - np.percentile(y, [25])

    r_q = np.sum(np.sign(x - med_x) * np.sign(y - med_y)) / n

    theta_1 = r_q * iqr_y / iqr_x
    theta_0 = med_y - theta_1 * med_x

    return theta_0, theta_1


def show_result(x, y, theta, theta_ethalon, criterion):
    print(criterion)
    print("Estimated coefficients:\nb_0 = {}  \
    \nb_1 = {}".format('%.5f' % theta[0], '%.5f' % theta[1]) + "\n")

    print("Ethalon coefficients:\nb_0 = {}  \
        \nb_1 = {}".format(theta_ethalon[0], theta_ethalon[1]))

    # plotting regression line
    plot_regression_line(x, y, theta)


def plot_regression_line(x, y, theta):
    plt.scatter(x, y, color="rosybrown")

    # predicted response vector
    y_pred = theta[0] + theta[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="teal")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2, 2.5)
    plt.title("n = 20")

    plt.show()


def main():
    x = np.linspace(-1.8, 2, n_size)
    y = ethalon_function(x)
    theta_ethalon = (2., 2.)

    for i in range(2):
        theta1 = estimate_least_squares(x, y)
        show_result(x, y, theta1, theta_ethalon, criterion="Least squares")

        theta2 = estimate_least_deviations(x, y)
        show_result(x, y, theta2, theta_ethalon, criterion="Least absolute deviations")

        # Add error in second iteration
        y[0] += 10.
        y[-1] -= 10.


if __name__ == "__main__":
    main()
