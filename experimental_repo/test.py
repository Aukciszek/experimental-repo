import copy
import os
import random

binary_internal = lambda n: n > 0 and [n & 1] + binary_internal(n >> 1) or []


def binary(n):
    if n == 0:
        return [0]
    else:
        return binary_internal(n)


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


def inverse_matrix_mod(matrix_dc, modulus):
    matrix_dc = copy.deepcopy(matrix_dc)

    n = len(matrix_dc)
    identity_matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    for i in range(n):
        # Find the first non-zero element in column i starting from row i
        non_zero_index = next(
            (k for k in range(i, n) if matrix_dc[k][i] % modulus != 0), -1
        )
        if non_zero_index == -1:
            raise ValueError("Matrix is not invertible under this modulus.")

        # Swap rows i and non_zero_index in both matrix and identity_matrix
        matrix_dc[i], matrix_dc[non_zero_index] = (
            matrix_dc[non_zero_index],
            matrix_dc[i],
        )
        identity_matrix[i], identity_matrix[non_zero_index] = (
            identity_matrix[non_zero_index],
            identity_matrix[i],
        )

        # Normalize the pivot row
        pivot = matrix_dc[i][i] % modulus
        pivot_inv = modular_multiplicative_inverse(pivot, modulus)

        matrix_dc[i] = [(x * pivot_inv) % modulus for x in matrix_dc[i]]
        identity_matrix[i] = [(x * pivot_inv) % modulus for x in identity_matrix[i]]

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            if matrix_dc[j][i] % modulus != 0:
                factor = matrix_dc[j][i]
                matrix_dc[j] = [
                    (matrix_dc[j][k] - factor * matrix_dc[i][k]) % modulus
                    for k in range(n)
                ]
                identity_matrix[j] = [
                    (identity_matrix[j][k] - factor * identity_matrix[i][k]) % modulus
                    for k in range(n)
                ]

    # Back substitution to eliminate entries above the pivots
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            if matrix_dc[j][i] % modulus != 0:
                factor = matrix_dc[j][i]
                matrix_dc[j] = [
                    (matrix_dc[j][k] - factor * matrix_dc[i][k]) % modulus
                    for k in range(n)
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
    return sum([coefficients[i] * x**i for i in range(t)]) % p


def Shamir(t, n, k0, p):

    coefficients = [random.randint(0, p - 1) for _ in range(t)]
    coefficients[0] = k0

    if coefficients[-1] == 0:
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
                li %= p
        coefficients.append(li)

    return coefficients

def reconstruct_secret(shares, coefficients, p):
    secret = 0

    for i, (_, y_i) in enumerate(shares):
        secret += y_i * coefficients[i]
        secret %= p

    return secret

###

def multiply_bit_shares_by_powers_of_2(shares):
    multiplied_shares = []
    for i in range(len(shares)):
        multiplied_shares.append((i,2**i * shares[i][1]))
    return multiplied_shares

def add_multiplied_shares(multiplied_shares):
    share_r = multiplied_shares[0][1]
    for i in range(1, len(multiplied_shares)):
        share_r += multiplied_shares[i][1]
    return share_r

###

s = 2
d = 3
# liczba bitów p = k+l
k = 1
# liczba bitów d,s <= l
l = 2
# liczba serwerow
n = 5
# serwery odzyskujace sekret
t = 2
# liczba pierwsza
p = 7


tab = [1,0,0,0,1]
print(tab)

shery_bitu_nr_0 = Shamir(t,n,tab[0],p)
shery_bitu_nr_1 = Shamir(t,n,tab[1],p)
shery_bitu_nr_2 = Shamir(t,n,tab[2],p)
shery_bitu_nr_3 = Shamir(t,n,tab[3],p)
shery_bitu_nr_4 = Shamir(t,n,tab[4],p)

print(shery_bitu_nr_0)

shery_dla_party_1 = [shery_bitu_nr_0[0],shery_bitu_nr_1[0],shery_bitu_nr_2[0],shery_bitu_nr_3[0],shery_bitu_nr_4[0]]
shery_dla_party_2 = [shery_bitu_nr_0[1],shery_bitu_nr_1[1],shery_bitu_nr_2[1],shery_bitu_nr_3[1],shery_bitu_nr_4[1]]
shery_dla_party_3 = [shery_bitu_nr_0[2],shery_bitu_nr_1[2],shery_bitu_nr_2[2],shery_bitu_nr_3[2],shery_bitu_nr_4[2]]
shery_dla_party_4 = [shery_bitu_nr_0[3],shery_bitu_nr_1[3],shery_bitu_nr_2[3],shery_bitu_nr_3[3],shery_bitu_nr_4[3]]
shery_dla_party_5 = [shery_bitu_nr_0[4],shery_bitu_nr_1[4],shery_bitu_nr_2[4],shery_bitu_nr_3[4],shery_bitu_nr_4[4]]

print(shery_dla_party_1)

pomnozone_shery_dla_party_1 = multiply_bit_shares_by_powers_of_2(shery_dla_party_1)
pomnozone_shery_dla_party_2 = multiply_bit_shares_by_powers_of_2(shery_dla_party_2)
pomnozone_shery_dla_party_3 = multiply_bit_shares_by_powers_of_2(shery_dla_party_3)
pomnozone_shery_dla_party_4 = multiply_bit_shares_by_powers_of_2(shery_dla_party_4)
pomnozone_shery_dla_party_5 = multiply_bit_shares_by_powers_of_2(shery_dla_party_5)

print(pomnozone_shery_dla_party_1)

sher_r_party_1 = (1,add_multiplied_shares(pomnozone_shery_dla_party_1))
sher_r_party_2 = (2,add_multiplied_shares(pomnozone_shery_dla_party_2))
sher_r_party_3 = (3,add_multiplied_shares(pomnozone_shery_dla_party_3))
sher_r_party_4 = (4,add_multiplied_shares(pomnozone_shery_dla_party_4))
sher_r_party_5 = (5,add_multiplied_shares(pomnozone_shery_dla_party_5))

print(sher_r_party_1)

selected_shares = [shery_dla_party_1[0],shery_dla_party_2[0],shery_dla_party_3[0],shery_dla_party_4[0],shery_dla_party_5[0]]
coefficients = computate_coefficients(selected_shares, p)
secret = reconstruct_secret(selected_shares, coefficients, p)
print("reconstructed bit nr  0: ", secret % p)

selected_shares = [sher_r_party_1,sher_r_party_2,sher_r_party_3,sher_r_party_4,sher_r_party_5]
coefficients = computate_coefficients(selected_shares, p)
secret = reconstruct_secret(selected_shares, coefficients, p)
print("reconstructed r: ", secret % p)
