from numpy import exp

hbar = 1.054_571_628e-34  # reducted const of Planck (J.s)
m = 9.1e-31  # mass of electron (kg)


def GaussWP(
    k0: float,
    amp: float,
    x: float,
    t: float
) -> complex:
    omega = hbar * pow(k0, 2) / (2 * m)
    return amp * exp(1j * (k0 * x - omega * t))


if __name__ == "__main__":
    k0 = 2
    a = 2
    x = 2
    t = 0

    print(GaussWP(k0, a, x, t))
