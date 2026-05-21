from numpy import arange, ndarray


def power_function(range_: ndarray, degree: float):
    array_ = []

    for i in range_:
        array_.append(pow(i, degree))
    return array_


def derive(values: list, i: int) -> float:
    if i >= len(values):
        raise ValueError("The index is higher than the length of this array")

    if i == 0 or i == len(values) - 1:
        raise ValueError("Values are missing to the left or the right of this index to derive")

    x_a = i - 1
    x_b = i + 1
    y_a = values[x_a]
    y_b = values[x_b]

    return (y_b - y_a) / (x_b - x_a)


if __name__ == "__main__":
    range_ = arange(1, 100, 0.1)
    degree = 5

    ftn_list = power_function(range_, degree)
    print(derive(ftn_list, 100))
