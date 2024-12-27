import os
import random


def binary_exponentiation(b, k, n):
    a = 1
    while k:
        if k & 1:
            a = (a * b) % n
        b = (b * b) % n
        k >>= 1
    return a


def get_random_bits(bit_length: int) -> int:
    return int.from_bytes(os.urandom((bit_length + 7) // 8), "big")


def get_power_2_factors(n):
    r = 0

    if n <= 0:
        return 0, 0

    while n % 2 == 0:
        n = n // 2
        r += 1

    return r, n


def MillerRabin_prime_test(n, k):
    r, d = get_power_2_factors(n - 1)

    for _ in range(k):
        a = get_random_bits(n.bit_length())

        while a not in range(2, n - 2 + 1):
            a = get_random_bits(n.bit_length())

        x = binary_exponentiation(a, d, n)

        if x == 1 or x == n - 1:
            continue

        n_1_found = False

        for _ in range(r - 1):
            x = binary_exponentiation(x, 2, n)

            if x == n - 1:
                n_1_found = True
                break

        if not n_1_found:
            return False

    return True


def f(x, coefficients, p, t):
    return sum([coefficients[i] * x**i for i in range(t)]) % p


def Shamir(t, n, k0):
    p = 21

    coefficients = [random.randint(0, p - 1) for i in range(t)]  # TODO:urandom
    coefficients[0] = k0

    if coefficients[-1] == 0:
        coefficients[-1] = random.randint(1, p - 1)  # TODO: urandom

    shares = []

    for i in range(1, n + 1):
        shares.append((i, f(i, coefficients, p, t)))

    return shares, p


def computate_coefficients(shares, t):
    coefficients = [1] * t

    for i in range(t):
        x_i, _ = shares[i]

        for j in range(t):
            if i != j:
                x_j, _ = shares[j]

                coefficients[i] *= -x_j / (x_i - x_j)

    return coefficients


def reconstruct_secret(shares, coefficients, t):
    secret = 0

    for i in range(t):
        _, y_i = shares[i]

        secret += y_i * coefficients[i]

    return secret


def main():
    t = 2
    n = 2
    k0 = 4

    shares, p = Shamir(t, n, k0)

    print("shares = ", shares)
    print("p = ", p)

    shares = [(1, 20), (2, 15)]

    coefficients = computate_coefficients(shares, t)

    print("coefficients = ", coefficients)

    secret = reconstruct_secret(shares, coefficients, t)

    print("secret = ", secret % p)

    assert k0 == round(secret % p)


if __name__ == "__main__":
    main()
