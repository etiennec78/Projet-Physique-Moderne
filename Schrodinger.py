from numpy import empty, ndarray, linspace

from Const import hbar, m
from Derivation import derive
from PaquetOndeGauss1d4J import GaussWP


def waveFunction(
    nx: int, nt: int, L: float, T: float, k0: float, a: float, V0: float
) -> ndarray:
    """Initialize and fill a packet function as a 2D table

    Params:
    nx: amount of space points
    nt: amount of time points
    L: total length of the space interval
    T: total duration of the simulation
    k0: initial wave number
    a: width of the wave packet
    V0: potential energy

    Returns: The wave function as a numpy array [nt][nx]
    """

    # Initialize and fill in the wave function table
    wave_table, space_table, time_table = initWaveFunction(
        nx, nt, L, T, k0, a
    )
    wave_table = completeWaveFunction(wave_table, space_table, time_table, V0)

    return wave_table


def initWaveFunction(
    nx: int, nt: int, L: float, T: float, k0: float, a: float
) -> (ndarray, ndarray, ndarray):
    """Initialize a wave packet function as a 2D table at t=0.

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


def completeWaveFunction(
    wave_table: ndarray, x_tab: ndarray, t_tab: ndarray, V0: float
) -> ndarray:
    """Complete the wave function table by calculating the values at t>0.

    Params:
    wave_table: the 2D table containing the wave function values
    x_tab: the list of spacial values
    t_tab: the list of time values
    V0: potential energy

    Returns: The 2D table containing the wave function values
    """

    # Get the length of time and position axis
    nt, nx = wave_table.shape

    # Set the edge values to 0 since we cannot derive them
    wave_table[:, 0:2] = 0
    wave_table[:, -2:] = 0

    # Get the size of dt
    dt = t_tab[1] - t_tab[0]

    # For each time line
    for n in range(0, nt - 1):
        current_y = wave_table[n, :]

        # For each position
        for j in range(2, nx - 2):
            # Get the second position derivative
            d2psi_dx2 = derive(x_tab, current_y, j, 2)

            # 2. Schrödinger equation
            kinetic = -(hbar**2 / (2 * m)) * d2psi_dx2
            potential = V0 * wave_table[n, j]
            dpsi_dt = (kinetic + potential) / (1j * hbar)

            # 3. Euler method to get the next position
            wave_table[n + 1, j] = wave_table[n, j] + dt * dpsi_dt

    return wave_table


if __name__ == "__main__":
    K = 5e9
    A = 1.5e-8

    NX = 500
    NT = 2000
    LENGTH = 80e-9
    DURATION = 1e-15

    wave_table, space_table, time_table = initWaveFunction(
        NX, NT, LENGTH, DURATION, K, A
    )
