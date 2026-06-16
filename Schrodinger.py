from numpy import empty, ndarray, linspace

from PaquetOndeGauss1d4J import GaussWP


def wave_function(
    nx: int, nt: int, L: float, T: float, k0: float, a: float
) -> ndarray:
    """A wave function as a 2D table.

    Params:
    nx: amount of space points
    nt: amount of time points
    L: total length of the space interval
    T: total duration of the simulation
    k0: initial wave number
    a: width of the wave packet

    Returns:
    * The wave function as a numpy array [nt][nx]
    * The list of spacial values
    * The list of time values
    """
    if nx < 1 or nt < 1:
        raise ValueError("The amount of lines and columns must be at least 1.")

    # Generate lists containing x and t values
    x_tab = linspace(-L / 2, L / 2, nx)
    t_tab = linspace(0, T, nt)

    # Generate a 2D tab containing random data
    wave_table = empty((nt, nx), dtype=complex)  # Initialize as complex for WP

    # Fill in values in the first line (t=0)
    for j, x in enumerate(x_tab):
        wave_table[0, j] = GaussWP(k0, a, x, 0)

    return wave_table, x_tab, t_tab


if __name__ == "__main__":
    K = 5e9
    A = 1.5e-8

    NX = 500
    NT = 2000
    LENGTH = 80e-9
    DURATION = 1e-15

    wave_table, space_table, time_table = wave_function(NX, NT, LENGTH, DURATION, K, A)
