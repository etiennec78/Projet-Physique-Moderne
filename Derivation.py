from numpy import arange


def derive(x_values: list, y_values: list, i: int) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("Both lists should have the same length")

    if i >= len(x_values):
        raise ValueError("The index is higher than the length of this array")

    if i == 0 or i == len(x_values) - 1:
        raise ValueError(
            "Values are missing to the left or the right of this index to derive"
        )

    a = i - 1
    b = i + 1

    x_a = x_values[a]
    x_b = x_values[b]
    y_a = y_values[a]
    y_b = y_values[b]

    return (y_b - y_a) / (x_b - x_a)


def second_derivative(x_values: list, y_values: list, i: int) -> float:
    a = i - 1
    b = i + 1

    y_a = derive(x_values, y_values, a)
    y_b = derive(x_values, y_values, b)

    x_a = x_values[a]
    x_b = x_values[b]

    return (y_b - y_a) / (x_b - x_a)


if __name__ == "__main__":
    # Deriving x²
    x_values = arange(1, 100, 0.1)
    y_values = x_values**2
    print(derive(x_values, y_values, 100))

    # Double deriving x²
    x_values = arange(1, 100, 0.1)
    print(second_derivative(x_values, y_values, 100))

    # Deriving 2x
    y_values = 2 * x_values
    print(derive(x_values, y_values, 100))
