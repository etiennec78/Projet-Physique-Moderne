from numpy import exp, pi

hbar = 1.054_571_628e-34  # reducted const of Planck (J.s)
m = 9.1e-31  # mass of electron (kg)


def GaussWP(
    k0: float,
    a: float,
    x: float,
    t: float
) -> complex:
    amp = pow(2 / (pi * pow(a, 2)), 1 / 4)
    omega = hbar * pow(k0, 2) / (2 * m)

    envelop = exp(-pow(x, 2) / pow(a, 2))
    plane_wave = exp(1j * (k0 * x - omega * t))

    return amp * envelop * plane_wave


if __name__ == "__main__":
    k0 = 2
    a = 2
    x = 2
    t = 0

    print(GaussWP(k0, a, x, t))
