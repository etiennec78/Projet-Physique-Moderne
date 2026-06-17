from matplotlib import pyplot as plt
from numpy import argmax, empty, ndarray, linspace

from Const import hbar, m
from Derivation import derive
from PaquetOndeGauss1d4J import GaussWP


class waveFunction:
    """Class to create a wave function table and fill it with values"""

    _wave_table: ndarray
    _x_tab: ndarray
    _t_tab: ndarray

    def __init__(
        self,
        nx: int,
        nt: int,
        L: float,
        T: float,
        k0: float,
        a: float,
        V0: float | ndarray,
    ) -> None:
        """Initialize and fill a packet function as a 2D table

        Params:
        nx: amount of space points
        nt: amount of time points
        L: total length of the space interval
        T: total duration of the simulation
        k0: initial wave number
        a: width of the wave packet
        V0: potential energy (a float constant or function)
        """
        self._wave_table, self._x_tab, self._t_tab = self._initWaveFunction(
            nx, nt, L, T, k0, a
        )
        self._completeWaveFunction(V0)

    def _initWaveFunction(
        self, nx: int, nt: int, L: float, T: float, k0: float, a: float
    ) -> None:
        """Initialize a wave packet function as a 2D table at t=0.

        Params:
        nx: amount of space points
        nt: amount of time points
        L: total length of the space interval
        T: total duration of the simulation
        k0: initial wave number
        a: width of the wave packet
        """
        if nx < 1 or nt < 1:
            raise ValueError("The amount of lines and columns must be at least 1.")

        # Generate lists containing x and t values
        x_tab = linspace(-L / 2, L / 2, nx)
        t_tab = linspace(0, T, nt)

        # Generate a 2D tab containing random data
        __wave_table = empty((nt, nx), dtype=complex)  # Initialize as complex for WP

        # Fill in values in the first line (t=0)
        for j, x in enumerate(x_tab):
            __wave_table[0, j] = GaussWP(k0, a, x, 0)

        return __wave_table, x_tab, t_tab

    def _completeWaveFunction(self, V0: float | ndarray) -> ndarray:
        """Complete the wave function table by calculating the values at t>0.

        Params:
        V0: potential energy (a float constant or function)
        """
        V0_is_array = isinstance(V0, ndarray)

        # Get the length of time and position axis
        nt, nx = self._wave_table.shape

        # Set the edge values to 0 since we cannot derive them
        self._wave_table[:, 0:2] = 0
        self._wave_table[:, -2:] = 0

        # Get the size of dt
        dt = self._t_tab[1] - self._t_tab[0]

        # For each time line
        for n in range(0, nt - 1):
            current_y = self._wave_table[n, :]

            # For each position
            for j in range(2, nx - 2):
                # Get the second position derivative
                d2psi_dx2 = derive(self._x_tab, current_y, j, 2)

                # Get the value of V0
                V0_value = V0[j] if V0_is_array else V0

                # Schrödinger equation
                kinetic = -(hbar**2 / (2 * m)) * d2psi_dx2
                potential = V0_value * self._wave_table[n, j]
                dpsi_dt = (kinetic + potential) / (1j * hbar)

                # Euler method to get the next position
                self._wave_table[n + 1, j] = self._wave_table[n, j] + dt * dpsi_dt

        return self._wave_table

    def plot(self) -> None:
        """Plot the wave function."""
        fig, ax = plt.subplots()
        for i in range(len(self._t_tab)):
            ax.plot(self._x_tab, abs(self._wave_table[i]))
        plt.show()

    def calculate_travel_time(self, distance: float) -> float | None:
        """Calculate the time it takes for a particule to travel a distance."""

        # Find the initial position (maximum probability at t=0)
        initial_prob = abs(self._wave_table[0]) ** 2
        start_idx = argmax(initial_prob)
        start_x = self._x_tab[start_idx]

        # Set the target
        target_x = start_x + distance$

        # Travel the time to see when the maximum passes by the target
        for i in range(len(self._t_tab)):
            current_prob = abs(self._wave_table[i]) ** 2
            current_idx = argmax(current_prob)
            current_x = self._x_tab[current_idx]

            if current_x >= target_x:
                return self._t_tab[i]

        # If the particule has not reached the destination
        return None


if __name__ == "__main__":
    K = 5e9
    A = 1.5e-8
    V0 = 0

    NX = 500
    NT = 2000
    LENGTH = 80e-9
    DURATION = 1e-15
    TARGET_DISTANCE = 10e-9

    wave_function = waveFunction(NX, NT, LENGTH, DURATION, K, A, V0)
    print(wave_function.calculate_travel_time(12))
    wave_function.plot()
