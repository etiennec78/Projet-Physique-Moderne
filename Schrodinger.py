from matplotlib import pyplot as plt
from numpy import argmax, empty, ndarray, linspace, searchsorted

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
        V: float | ndarray,
    ) -> None:
        """Initialize and fill a packet function as a 2D table

        Params:
        nx: amount of space points
        nt: amount of time points
        L: total length of the space interval
        T: total duration of the simulation
        k0: initial wave number
        a: width of the wave packet
        V: potential energy (a float constant or function)
        """
        self._wave_table, self._x_tab, self._t_tab = self._initWaveFunction(
            nx, nt, L, T, k0, a
        )
        self._completeWaveFunction(V)

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

    def _completeWaveFunction(self, V: float | ndarray) -> ndarray:
        """Complete the wave function table by calculating the values at t>0.

        Params:
        V: potential energy (a float constant or function)
        """
        V_is_array = isinstance(V, ndarray)

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

                # Get the value of V
                V_value = V[j] if V_is_array else V

                # Schrödinger equation
                kinetic = -(hbar**2 / (2 * m)) * d2psi_dx2
                potential = V_value * self._wave_table[n, j]
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
        target_x = start_x + distance

        # Travel the time to see when the maximum passes by the target
        for i in range(len(self._t_tab)):
            current_prob = abs(self._wave_table[i]) ** 2
            current_idx = argmax(current_prob)
            current_x = self._x_tab[current_idx]

            if current_x >= target_x:
                return self._t_tab[i]

        # If the particule has not reached the destination
        return None

    def calculate_crossing_time(self, x_start: float, x_end: float) -> float | None:
        """Calculate the time to go through a barrier."""
        t_in = None
        t_out = None

        # Get the index of the closest value before x_end
        idx_end = searchsorted(self._x_tab, x_end)

        for i in range(len(self._t_tab)):
            proba = abs(self._wave_table[i])**2

            # Find the time of entry (t_in)
            if t_in is None:
                global_max_idx = argmax(proba)
                if self._x_tab[global_max_idx] >= x_start:
                    t_in = self._t_tab[i]

            # Find the time of exit (t_out)
            elif t_out is None:
                # Only keep the next values to avoid searching in the reflected values
                proba_transmitted = proba[idx_end:]

                if len(proba_transmitted) > 0:
                    local_max_idx = argmax(proba_transmitted)

                    # Filter out waves that did not go through yet and noise
                    if local_max_idx > 2 and proba_transmitted[local_max_idx] > 1e-10:
                        t_out = self._t_tab[i]
                        break

        # Calculate the time in the barrier
        if t_in is not None and t_out is not None:
            return t_out - t_in

        return None


if __name__ == "__main__":
    K = 5e9
    V0 = 0
    X_START_BAR = 10e-9
    X_END_BAR = 15e-9

    NX = 500
    NT = 2000
    LENGTH = 80e-9
    DURATION = 1e-15
    TARGET_DISTANCE = 10e-9

    A = X_END_BAR - X_START_BAR

    wave_function = waveFunction(NX, NT, LENGTH, DURATION, K, A, V0)
    print(wave_function.calculate_travel_time(12))
    print(wave_function.calculate_crossing_time(X_START_BAR, X_END_BAR))
    wave_function.plot()
