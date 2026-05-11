from matplotlib import pyplot as plt
from numpy import exp, real, arange, ndarray
from collections.abc import Callable


def PlaneWave(amp: float, k: float, omega: float, x: float, t: float) -> complex:
    """One-dimensional plane-wave.

    Params:
    amp: amplitude of the wave
    k: wave number (rad/m)
    omega: pulsation of the wave (rad/s)
    x: position on the x axis (m)
    t: time (s)

    Returns the complex vibration quantity of the wave.
    """
    return amp * exp(1j * (k * x - omega * t))


def PlotFunction(
    amp: float,
    k: float,
    omega: float,
    range_: ndarray,
    t: float,
    filter_: Callable = real,
) -> None:
    """Plotting of a one-dimensional plane-wave using matplotlib.

    Params:
    amp: amplitude of the wave
    k: wave number (rad/m)
    omega: pulsation of the wave (rad/s)
    range_: the range that will be used for values on the x axis (start, end, step)
    t: time (s)
    filter_: A function to filter the data (e.g.: real() or imag() from numpy).
    """
    fig, ax = plt.subplots()
    values = []
    for x in range_:
        y = PlaneWave(amp, k, omega, x, t)
        if filter_ is not None:
            y = filter_(y)
        values.append(y)
    ax.plot(list(range_), values)
    plt.show()


if __name__ == "__main__":
    PlotFunction(2, 2, 1, arange(1, 30, 0.05), 0)
