from numpy import arange, ndarray


def power_function(range_: ndarray, degree: float):
    array_ = []

    for i in range_:
        array_.append(pow(i, degree))
    return array_


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


if __name__ == "__main__":
    DEGREE = 5

    x_values = arange(1, 100, 0.1)
    y_values = power_function(x_values, DEGREE)

    print(derive(x_values, y_values, 100))
