from dataclasses import dataclass
from matplotlib import pyplot as plt
from numpy import arange, argmax, empty, ndarray, linspace, searchsorted, zeros

from Const import hbar, m
from Derivation import derive
from PaquetOndeGauss1d4J import GaussWP


@dataclass
class BarrierPotential:
    V0: float
    x_start: float
    a: float

    def to_array(self, x_tab: ndarray) -> ndarray:
        """Generate an ndarray from these potential values"""
        V_tab = zeros(len(x_tab))
        for i, x in enumerate(x_tab):
            if self.x_start <= x <= self.x_start + self.a:
                V_tab[i] = self.V0
        return V_tab


@dataclass
class WaveFunctionData:
    """A dataclass containing all data about a wave function.

    Params:
    nx: amount of space points
    nt: amount of time points
    L: total length of the space interval
    T: total duration of the simulation
    k0: initial wave number
    A: width of the wave packet
    V: potential energy dataclass
    """

    nx: int
    nt: int
    L: float
    T: float
    k0: float
    A: float
    V: BarrierPotential


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
        self._data = data

    def _initWaveFunction(
        self, data: WaveFunctionData
    ) -> tuple[ndarray, ndarray, ndarray]:
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
        _wave_table = empty(
            (data.nt, data.nx), dtype=complex
        )  # Initialize as complex for WP

        # Fill in values in the first line (t=0)
        for j, x in enumerate(x_tab):
            _wave_table[0, j] = GaussWP(data.k0, data.A, x, 0)

        return _wave_table, x_tab, t_tab

    def _completeWaveFunction(self, V: BarrierPotential) -> ndarray:
        """Complete the wave function table by calculating the values at t>0.

        Params:
        V: potential energy dataclass
        """

        V_array = V.to_array(self._x_tab)

        # Get the length of time and position axis
        nt, nx = self._wave_table.shape

        # Set the edge values to 0 since we cannot derive them
        self._wave_table[:, 0:2] = 0
        self._wave_table[:, -2:] = 0

        # Get the size of dt et dx
        dt = self._t_tab[1] - self._t_tab[0]
        dx = self._x_tab[1] - self._x_tab[0]

        # For each time line
        for n in range(0, nt - 1):
            current_y = self._wave_table[n, :]

            # For each position
            for j in range(2, nx - 2):
                # Get the second position derivative
                d2psi_dx2 = (
                    current_y[j + 1] - 2 * current_y[j] + current_y[j - 1]
                ) / dx**2

                V_value = V_array[j]

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
        ax_potential = ax.twinx()

        nt = len(self._t_tab)
        step = max(1, nt // 8)
        for i in range(0, nt, step):
            ax.plot(
                self._x_tab,
                abs(self._wave_table[i]),
                label=f"t = {self._t_tab[i]:.2e} s",
            )

        ax_potential.plot(self._x_tab, self._data.V.to_array(self._x_tab), label="Potentiel V(x)")
        ax_potential.set_ylabel("Potentiel V(x) (J)")

        ax.set_title("Évolution temporelle de la fonction d'onde")
        ax.set_xlabel("Position x (m)")
        ax.set_ylabel("Amplitude |ψ(x, t)|")
        ax.legend(loc="upper left")
        plt.show()

    def calculate_travel_time(self, distance: float) -> float | None:
        """Calculate the time it takes for a particule to travel a distance.

        Params:
        distance: the distance to travel (in meters)
        """

        # Find the initial position of the packet
        start_x = self._x_tab[argmax(abs(self._wave_table[0]) ** 2)]

        target_idx = searchsorted(self._x_tab, start_x + distance)
        if target_idx >= len(self._x_tab):
            return None

        return self._t_tab[argmax(abs(self._wave_table[:, target_idx]) ** 2)]

    def calculate_crossing_time(self, x_start: float, x_end: float) -> float | None:

        # identifies the beginning and the end of the barrier
        idx_start = searchsorted(self._x_tab, x_start)
        idx_end = searchsorted(self._x_tab, x_end)

        # identifies the time at which the max of the packet crosses these positions
        t_in = self._t_tab[argmax(abs(self._wave_table[:, idx_start]) ** 2)]
        t_out = self._t_tab[argmax(abs(self._wave_table[:, idx_end]) ** 2)]

        return t_out - t_in if t_out > t_in else None


def plot_attr_influence(
    data: WaveFunctionData,
    attribute: str,
    range_: ndarray,
    simulation_values: list | ndarray,
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
    values: list[float] = []

    for new_val in simulation_values:
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


def study_influence_of_length(nx, nt, L, T, k, A, V0, x_start, range_a):
    """Plot a graph to study the influence of the barrier length."""
    tau_0_vals, tau_t_vals, valid_a_0, valid_a_t = [], [], [], []

    for a_val in range_a:
        # Free particle (V = 0)
        data_free = WaveFunctionData(
            nx, nt, L, T, k, A, BarrierPotential(0.0, x_start, a_val)
        )
        t_0 = WaveFunction(data_free).calculate_travel_time(a_val)
        if t_0 is not None:
            tau_0_vals.append(t_0)
            valid_a_0.append(a_val)

        # With barrier (V = V0)
        data_barrier = WaveFunctionData(
            nx, nt, L, T, k, A, BarrierPotential(V0, x_start, a_val)
        )
        t_t = WaveFunction(data_barrier).calculate_crossing_time(
            x_start, x_start + a_val
        )
        if t_t is not None:
            tau_t_vals.append(t_t)
            valid_a_t.append(a_val)

    fig, ax = plt.subplots()
    ax.plot(valid_a_0, tau_0_vals, label=r"$\tau_{0,num}$ (Particule libre)")
    ax.plot(valid_a_t, tau_t_vals, label=r"$\tau_{t,num}$ (Franchissement)")
    ax.set_title("Influence de la longueur 'a' sur les temps de parcours")
    ax.set_xlabel("Longueur a (m)")
    ax.set_ylabel("Temps (s)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # --- Ploting a wave function ---

    K = 5e9
    V0 = 1.5 * 1.602176634e-19
    X_START_BAR = 10e-9
    X_END_BAR = 15e-9

    NX = 500
    NT = 40000
    LENGTH = 80e-9
    DURATION = 6e-14
    TARGET_DISTANCE = 5e-9

    a = X_END_BAR - X_START_BAR
    A = 1e-9

    V_barrier = BarrierPotential(V0, X_START_BAR, a)
    wave_data = WaveFunctionData(NX, NT, LENGTH, DURATION, K, A, V_barrier)
    wave_function = WaveFunction(wave_data)
    wave_function.plot()

    # --- Calculating travel & cross times ---
    travel_time = wave_function.calculate_travel_time(TARGET_DISTANCE)
    crossing_time = wave_function.calculate_crossing_time(X_START_BAR, X_END_BAR)
    print(f"Travel time: {travel_time}")
    print(f"Crossing time: {crossing_time}")

    # --- Influence of parameters ---

    # Influence of a on calculate_travel_time
    RANGE_START = 1e-9
    RANGE_END = 8e-9
    RANGE_STEP = 1e-9
    range_ = arange(RANGE_START, RANGE_END, RANGE_STEP)

    study_influence_of_length(
        NX, NT, LENGTH, DURATION, K, A, V0, X_START_BAR, range_
    )

    # Influence of V on calculate_crossing_time
    ATTRIBUTE = "V"
    METHOD = "calculate_crossing_time"

    RANGE_START = 0.25 * V0
    RANGE_END = 4.0 * V0
    RANGE_STEP = 0.25 * V0
    range_ = arange(RANGE_START, RANGE_END, RANGE_STEP)

    potential_range = [
        BarrierPotential(v0, X_START_BAR, a)
        for v0 in range_
    ]

    plot_attr_influence(
        wave_data, ATTRIBUTE, range_, potential_range, METHOD, X_START_BAR, X_END_BAR
    )
