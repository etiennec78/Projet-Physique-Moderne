from matplotlib import pyplot as plt
from numpy import arange, exp, imag, ndarray, pi, real

from Const import hbar, m


def GaussWP(k0: float, a: float, x: float, t: float) -> complex:
    amp = (2 / (pi * a**2)) ** (1 / 4)
    omega = hbar * k0**2 / (2 * m)

    envelop = exp(-(x**2) / a**2)
    plane_wave = exp(1j * (k0 * x - omega * t))

    return amp * envelop * plane_wave


def plotGaussWP(k0: float, a: float, t: float, range_: ndarray) -> None:
    fig, ax = plt.subplots()
    values_real = []
    values_imag = []

    for x in range_:
        y = GaussWP(k0, a, x, t)

        values_real.append(real(y))
        values_imag.append(imag(y))

    range_list = list(range_)
    ax.plot(range_list, values_real, label="Partie réelle")
    ax.plot(range_list, values_imag, label="Partie imaginaire")
    ax.set_title("Paquet d'onde gaussien")
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    K = 5e9
    A = 1.5e-8
    T = 0

    range_start = -40e-9
    range_end = 40e-9
    RANGE_STEP = 0.1e-9
    range_ = arange(range_start, range_end, RANGE_STEP)

    plotGaussWP(K, A, T, range_)
