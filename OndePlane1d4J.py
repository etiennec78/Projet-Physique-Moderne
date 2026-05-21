from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from numpy import exp, real, arange, ndarray, pi, cos
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


def plotWaves(
    amp: float,
    k: float,
    delta_k: float,
    omega: float,
    t: float,
    range_: ndarray,
    filter_: Callable,
    values: list(list(float)),
    sum_: list(float),
    ax: Axes,
) -> None:
    """Plot 3 plane waves.

    Params:
    amp: amplitude of the wave
    k: wave number (rad/m)
    delta_k: margin of k (rad/m)
    omega: pulsation of the wave (rad/s)
    t: time (s)
    range_: the range that will be used for values on the x axis (start, end, step)
    filter_: A function to filter the data (e.g.: real() or imag() from numpy)
    values: list of each wave y values
    sum_: list of sum of all waves y values
    ax: Axes of the plot
    """

    amps = [amp, amp / 2, amp / 2]
    k = [k, k - delta_k / 2, k + delta_k / 2]
    for i in range(3):
        values.append([])
        for x in range_:
            # Get the value of the wave
            y = PlaneWave(amps[i], k[i], omega, x, t)
            if filter_ is not None:
                y = filter_(y)

            values[i].append(y)

            # Sum the value of this wave with others
            if i == 0:
                sum_.append(y)
            else:
                sum_[len(values[i]) - 1] += y

        # Plot the current wave
        range_list = list(range_)
        ax.plot(range_list, values[i])


def plotEnvelop(
    amp: float,
    delta_k: float,
    range_: ndarray,
    filter_: Callable,
    values: list(list(float)),
    ax: Axes,
) -> None:
    """Plot 3 plane waves.

    Params:
    amp: amplitude of the wave
    delta_k: margin of k (rad/m)
    range_: the range that will be used for values on the x axis (start, end, step)
    filter_: A function to filter the data (e.g.: real() or imag() from numpy)
    values: list of each wave y values
    ax: Axes of the plot
    """

    # Get the current length of the table
    i = len(values)

    for j in range(i, i + 3):
        values.append([])
        for x in range_:
            # Get the value of the wave
            y = pow(-1, j) * amp * (1 + cos(delta_k / 2 * x))
            if filter_ is not None:
                y = filter_(y)
            values[j].append(y)
        range_list = list(range_)
        ax.plot(range_list, values[j])


def PlotFunction(
    amp: float,
    k: float,
    omega: float,
    delta_k: float,
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
    filter_: A function to filter the data (e.g.: real() or imag() from numpy)
    """
    fig, ax = plt.subplots()
    values = []
    sum_ = []

    plotWaves(amp, k, delta_k, omega, t, range_, filter_, values, sum_, ax)

    # Plot the wave representing the sum of waves
    ax.plot(list(range_), sum_)

    plotEnvelop(amp, delta_k, range_, filter_, values, ax)

    plt.show()


if __name__ == "__main__":
    AMP = 2
    K = 2
    OMEGA = 1
    T = 0
    DELTA_K = 0.5

    range_start = -pi / DELTA_K
    range_end = pi / DELTA_K
    RANGE_STEP = 0.05
    range_ = arange(range_start, range_end, RANGE_STEP)

    PlotFunction(AMP, K, OMEGA, DELTA_K, range_, T)
