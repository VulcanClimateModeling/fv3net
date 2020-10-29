import matplotlib.pyplot as plt
import numpy as np


def plot_compare_means(pred, dQ_test, y_test):
    sli, qt = np.split(pred, 2, axis=1)
    sli_true, qt_true = np.split(dQ_test, 2, axis=1)

    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

    y_test = y_test.astype(bool).ravel()

    def plot_var(ax, sli_true, sli):
        ax.plot(sli_true[y_test].mean(0), "b-", label="truth")
        ax.plot(sli[y_test].mean(0), "b--", label="prediction")
        ax.plot(sli_true[~y_test].mean(0), "k-")
        ax.plot(sli[~y_test].mean(0), "k--")

    fig.suptitle(f"Triggered (blue) and not (black) points")
    plot_var(a, sli_true, sli)
    a.set_ylabel("K/s")
    plot_var(b, qt_true, qt)
    b.set_ylabel("kg/kg/s")

    a.set_xlabel("level")
    b.set_xlabel("level")
    fig.legend()


def plot_r2(pred, dQ_test):

    sse = ((pred - dQ_test) ** 2).sum(0)
    ss = ((dQ_test - dQ_test.mean(0)) ** 2).sum(0)
    r2 = 1 - sse / ss
    sli, qt = np.split(r2, 2)
    plt.plot(sli, label="dQ1")
    plt.plot(qt, label="dQ2")
    plt.legend()
