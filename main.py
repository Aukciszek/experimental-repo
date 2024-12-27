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


def modular_multiplicative_inverse(b: int, n: int) -> int:
    A = n
    B = b
    U = 0
    V = 1
    while B != 0:
        q = A // B
        A, B = B, A - q * B
        U, V = V, U - q * V
    if U < 0:
        return U + n
    return U


def inverse_matrix_mod(matrix, modulus):
    n = len(matrix)
    identity_matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    for i in range(n):
        # Find the first non-zero element in column i starting from row i
        non_zero_index = next(
            (k for k in range(i, n) if matrix[k][i] % modulus != 0), -1
        )
        if non_zero_index == -1:
            raise ValueError("Matrix is not invertible under this modulus.")

        # Swap rows i and non_zero_index in both matrix and identity_matrix
        matrix[i], matrix[non_zero_index] = matrix[non_zero_index], matrix[i]
        identity_matrix[i], identity_matrix[non_zero_index] = (
            identity_matrix[non_zero_index],
            identity_matrix[i],
        )

        # Normalize the pivot row
        pivot = matrix[i][i] % modulus
        pivot_inv = modular_multiplicative_inverse(pivot, modulus)

        matrix[i] = [(x * pivot_inv) % modulus for x in matrix[i]]
        identity_matrix[i] = [(x * pivot_inv) % modulus for x in identity_matrix[i]]

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            if matrix[j][i] % modulus != 0:
                factor = matrix[j][i]
                matrix[j] = [
                    (matrix[j][k] - factor * matrix[i][k]) % modulus for k in range(n)
                ]
                identity_matrix[j] = [
                    (identity_matrix[j][k] - factor * identity_matrix[i][k]) % modulus
                    for k in range(n)
                ]

    # Back substitution to eliminate entries above the pivots
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            if matrix[j][i] % modulus != 0:
                factor = matrix[j][i]
                matrix[j] = [
                    (matrix[j][k] - factor * matrix[i][k]) % modulus for k in range(n)
                ]
                identity_matrix[j] = [
                    (identity_matrix[j][k] - factor * identity_matrix[i][k]) % modulus
                    for k in range(n)
                ]

    return identity_matrix


def multiply_matrix(matrix1, matrix2, modulus):
    n = len(matrix1)
    m = len(matrix2[0])
    l = len(matrix2)

    if len(matrix1[0]) != l:
        raise ValueError("Matrix dimensions do not match.")

    result = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            result[i][j] = (
                sum(matrix1[i][k] * matrix2[k][j] % modulus for k in range(l)) % modulus
            )

    return result


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

    matrix = [[2, 4], [3, 5]]
    modulus = 7

    inverse = inverse_matrix_mod(matrix, modulus)
    print(inverse)

    matrix1 = [[1, 2], [3, 4]]
    matrix2 = [[5, 6], [7, 8]]

    result = multiply_matrix(matrix1, matrix2, 7)
    print(result)


if __name__ == "__main__":
    main()
