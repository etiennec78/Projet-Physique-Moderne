from matplotlib import pyplot as plt
from numpy import arange, exp, imag, ndarray, pi, real

hbar = 1.054_571_628e-34  # reducted const of Planck (J.s)
m = 9.1e-31  # mass of electron (kg)


def GaussWP(
    k0: float,
    a: float,
    x: float,
    t: float
) -> complex:
    amp = pow(2 / (pi * pow(a, 2)), 1 / 4)
    omega = hbar * pow(k0, 2) / (2 * m)

    envelop = exp(-pow(x, 2) / pow(a, 2))
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
    ax.plot(range_list, values_real)
    ax.plot(range_list, values_imag)
    plt.show()


if __name__ == "__main__":
    K = 2
    A = 2
    T = 0
    DELTA_K = 0.5

    range_start = -pi / DELTA_K
    range_end = pi / DELTA_K
    RANGE_STEP = 0.05
    range_ = arange(range_start, range_end, RANGE_STEP)

    plotGaussWP(K, A, T, range_)
