import random


def binary_exponentiation(b, k, n):
    if k < 0:
        k = n - 2

    a = 1
    while k:
        if k & 1:
            a = (a * b) % n
        b = (b * b) % n
        k >>= 1
    return a


def f(x, coefficients, p, t):
    return sum([coefficients[i] * x ** i for i in range(t)]) % p


def Shamir(t, n, k0, p):
    coefficients = [random.randint(0, p - 1) for _ in range(t)]
    coefficients[0] = k0

    # if t=1, the first coefficient is 0, as the polynomial is a horizontal line
    if coefficients[-1] == 0 and len(coefficients) > 1:
        coefficients[-1] = random.randint(1, p - 1)

    shares = []
    for i in range(1, n + 1):
        shares.append((i, f(i, coefficients, p, t)))

    return shares


def computate_coefficients(shares, p):
    coefficients = []

    for i, (x_i, _) in enumerate(shares):
        li = 1
        for j, (x_j, _) in enumerate(shares):
            if i != j:
                li *= x_j * binary_exponentiation(x_j - x_i, -1, p)
                # li %= p
        coefficients.append(li)

    return coefficients


def reconstruct_secret(shares, coefficients, p):
    secret = 0

    for i, (_, y_i) in enumerate(shares):
        secret += y_i * coefficients[i]
        # secret %= p

    return secret
