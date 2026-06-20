from dataclasses import dataclass
from matplotlib import pyplot as plt
from numpy import arange, argmax, empty, ndarray, linspace, searchsorted, zeros

from Const import hbar, m
from Derivation import derive
from PaquetOndeGauss1d4J import GaussWP


@dataclass
class WaveFunctionData:
    """A dataclass containing all data about a wave function.

    Params:
    nx: amount of space points
    nt: amount of time points
    L: total length of the space interval
    T: total duration of the simulation
    k0: initial wave number
    a: width of the wave packet
    V: potential energy (a float constant or function)
    """

    nx: int
    nt: int
    L: float
    T: float
    k0: float
    a: float
    V: float | ndarray


class WaveFunction:
    """Class to create a wave function table and fill it with values"""

    _data: WaveFunctionData
    _wave_table: ndarray
    _x_tab: ndarray
    _t_tab: ndarray

    def __init__(self, data: WaveFunctionData) -> None:
        """Initialize and fill a packet function as a 2D table"""
        self._wave_table, self._x_tab, self._t_tab = self._initWaveFunction(data)
        self._completeWaveFunction(data.V)

    def _initWaveFunction(self, data: WaveFunctionData) -> (ndarray, ndarray, ndarray):
        """Initialize a wave packet function as a 2D table at t=0.

        Params:
        data: A dataclass containing all wave data

        Returns:
        * The wave function as a numpy array [nt][nx]
        * The list of spacial values
        * The list of time values
        """
        if data.nx < 1 or data.nt < 1:
            raise ValueError("The amount of lines and columns must be at least 1.")

        # Generate lists containing x and t values
        x_tab = linspace(-data.L / 2, data.L / 2, data.nx)
        t_tab = linspace(0, data.T, data.nt)

        # Generate a 2D tab containing random data
        __wave_table = empty(
            (data.nt, data.nx), dtype=complex
        )  # Initialize as complex for WP

        # Fill in values in the first line (t=0)
        for j, x in enumerate(x_tab):
            __wave_table[0, j] = GaussWP(data.k0, data.a, x, 0)

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

        nt = len(self._t_tab)
        step = max(1, nt // 8)
        for i in range(0, nt, step):
            ax.plot(
                self._x_tab,
                abs(self._wave_table[i]),
            )

        ax.axvspan(10e-9, 15e-9, color="grey", alpha=0.3, label="Barrière")

        ax.set_title("Évolution temporelle de la fonction d'onde")
        ax.set_xlabel("Position x (m)")
        ax.set_ylabel("Amplitude |ψ(x, t)|")
        ax.legend()
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
            proba = abs(self._wave_table[i]) ** 2

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


def create_potential_barrier(
    nx: int, L: float, V0: float, x_start: float, x_end: float
) -> ndarray:
    """Create an array representing the potential function V(x).

    Params:
    nx: amount of space points
    L: total length of the space interval
    V0: potential energy
    x_start: The start of the barrier
    x_end: The end of the barrier

    Returns:
    The V(X) array
    """
    x_tab = linspace(-L / 2, L / 2, nx)
    V_tab = zeros(nx)
    for i, x in enumerate(x_tab):
        if x_start <= x <= x_end:
            V_tab[i] = V0
    return V_tab


def plot_attr_influence(
    data: WaveFunctionData,
    attribute: str,
    range_: ndarray,
    method_name: str,
    *args: list,
):
    """Plot the influence of a parameter on the wave function

    Parameters:
    data: The dataclass of the wave function
    attribute: The targeted attribute to vary
    range_: The range of values that the attribute will take
    method_name: The wave function method to call to get the y value
    *args: The args to give to the method on call
    """
    values: list(float) = []

    for new_val in range_:
        # Replace targeted attribute with values from the range
        setattr(data, attribute, new_val)
        wave_function = WaveFunction(data)

        # Apply the requested method and store results
        method = getattr(wave_function, method_name)
        value = method(*args)
        values.append(value)

    # Plot a graph showing influence of this attribute
    fig, ax = plt.subplots()
    valid_x = []
    valid_y = []
    for i, value in enumerate(values):
        # Don't draw the point if the value returned is None
        if value is not None:
            valid_x.append(range_[i])
            valid_y.append(value)

    ax.plot(valid_x, valid_y)

    # Set the legend
    ax.set_title(f"Influence du paramètre {attribute} sur la fonction {method_name}")
    ax.set_xlabel(f"Valeur de {attribute}")
    ax.set_ylabel(f"Valeur retournée par {method_name}")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    # --- Ploting a wave function ---

    K = 5e9
    V0 = 1.5 * 1.602176634e-19
    X_START_BAR = 10e-9
    X_END_BAR = 15e-9

    NX = 500
    NT = 6000
    LENGTH = 80e-9
    DURATION = 2.2e-14
    TARGET_DISTANCE = 10e-9

    A = X_END_BAR - X_START_BAR

    V_barrier = create_potential_barrier(NX, LENGTH, V0, X_START_BAR, X_END_BAR)
    wave_data = WaveFunctionData(NX, NT, LENGTH, DURATION, K, A, V_barrier)
    wave_function = WaveFunction(wave_data)
    wave_function.plot()

    # --- Calculating travel & cross times ---

    print(wave_function.calculate_travel_time(TARGET_DISTANCE))
    print(wave_function.calculate_crossing_time(X_START_BAR, X_END_BAR))

    # --- Influence of parameters ---

    RANGE_START = A
    RANGE_END = 2 * A
    RANGE_STEP = A / 10
    range_ = arange(RANGE_START, RANGE_END, RANGE_STEP)

    # Influence of a on calculate_travel_time
    ATTRIBUTE = "a"
    METHOD = "calculate_travel_time"

    RANGE_START = A
    RANGE_END = 2 * A
    RANGE_STEP = A / 10
    range_ = arange(RANGE_START, RANGE_END, RANGE_STEP)

    plot_attr_influence(
        wave_data, ATTRIBUTE, range_, METHOD, TARGET_DISTANCE
    )

    # Influence of V on calculate_crossing_time
    ATTRIBUTE = "V"
    METHOD = "calculate_crossing_time"

    RANGE_START = V0
    RANGE_END = 2 * V0
    RANGE_STEP = V0 / 10
    range_ = arange(RANGE_START, RANGE_END, RANGE_STEP)
    potential_range = [
        create_potential_barrier(NX, LENGTH, v0, X_START_BAR, X_END_BAR)
        for v0 in range_
    ]

    plot_attr_influence(
        wave_data, ATTRIBUTE, potential_range, METHOD, X_START_BAR, X_END_BAR
    )
