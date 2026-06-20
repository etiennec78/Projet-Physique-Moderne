from numpy import arange, ndarray


def derive(
    x_values: ndarray,
    y_values: ndarray,
    i: int,
    degree: int = 1,
) -> float | complex:
    """Approximate the derivative of a discrete function at one index.

    Params:
    x_values: sample positions
    y_values: sample values at each position
    i: index of the point to derive
    degree: derivative order

    Returns:
    The estimated derivative value at index i.
    """
    if len(x_values) != len(y_values):
        raise ValueError("Both lists should have the same length")

    if i >= len(x_values):
        raise ValueError("The index is higher than the length of this array")

    if i == 0 or i == len(x_values) - 1:
        raise ValueError(
            "Values are missing to the left or the right of this index to derive"
        )

    if degree == 0:
        return y_values[i]

    # Get the index of previous and next point
    a = i - 1
    b = i + 1

    # Get the associated x values
    x_a = x_values[a]
    x_b = x_values[b]

    # Get the associated y values
    if degree == 1:
        y_a = y_values[a]
        y_b = y_values[b]
    else:
        y_a = derive(x_values, y_values, a, degree - 1)
        y_b = derive(x_values, y_values, b, degree - 1)

    return (y_b - y_a) / (x_b - x_a)


if __name__ == "__main__":
    # Deriving x²
    x_values = arange(1, 100, 0.1)
    y_values = x_values**2
    derivative = derive(x_values, y_values, 100)
    print(f"The derivative of x² is {derivative}")

    # Double deriving x²
    x_values = arange(1, 100, 0.1)
    derivative = derive(x_values, y_values, 100, 2)
    print(f"The double derivative of x² is {derivative}")

    # Deriving 2x
    y_values = 2 * x_values
    derivative = derive(x_values, y_values, 100)
    print(f"The derivative of 2x is {derivative}")
