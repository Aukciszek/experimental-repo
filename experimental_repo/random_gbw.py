import copy
import os
import random

# random.seed(2137)
# print("Seed test: ",random.randint(0,100))

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


class Party:
    def __init__(self, t, n, id, p):
        self.__t = t
        self.__n = n
        self.__id = id
        self.__client_shares = []
        self.__p = p
        self.__parties = None
        self.__A = None
        self.__q = None # to co sam losuje
        self.__shared_q = [None] * n    # to co dostaje od innych
        self.__r = None
        self.__shared_r = [None] * n
        self.__multiplicative_share = None

    def set_shares(self, client_id, share):
        self.__client_shares.append((client_id, share))

    def set_parties(self, parties):
        if self.__parties is not None:
            raise ValueError("Parties already set.")

        if len(parties) != self.__n:
            raise ValueError("Invalid number of parties.")

        self.__parties = parties

    def calculate_A(self):
        if self.__A is not None:
            raise ValueError("A already calculated.")

        B = [list(range(1, self.__n + 1)) for _ in range(self.__n)]

        for j in range(self.__n):
            for k in range(self.__n):
                B[j][k] = binary_exponentiation(B[j][k], j, self.__p)

        B_inv = inverse_matrix_mod(B, self.__p)

        P = [[0] * self.__n for _ in range(self.__n)]

        for i in range(self.__t):
            P[i][i] = 1

        self.__A = multiply_matrix(multiply_matrix(B_inv, P, self.__p), B, self.__p)
    
    def calculate_q(self):
        if self.__q is not None:
            raise ValueError("q already calculated.")

        self.__q = [0] * self.__n
        self.__q, _ = Shamir(2*self.__t, self.__n, k0=0)

    # set q to other parties
    def _set_q(self, party_id, shared_q):
        if self.__shared_q[party_id - 1] is not None:
            raise ValueError("q already set.")

        self.__shared_q[party_id - 1] = shared_q

    # send q to other parties
    def send_q(self):
        for i in range(self.__n):
            if i == self.__id - 1:
                self.__shared_q[i] = self.__q[i]
                continue

            self.__parties[i]._set_q(self.__id, self.__q[i])

    def calculate_r(self, first_client_id, second_client_id):
        if self.__r is not None:
            raise ValueError("r already calculated.")

        self.__r = [0] * self.__n

        first_client_share = next(
            (y for x, y in self.__client_shares if x == first_client_id), None
        )
        second_client_share = next(
            (y for x, y in self.__client_shares if x == second_client_id), None
        )

        # receive q from other parties
        # add sum of qs in multiplied shares
        qs = [x[1] for x in self.__shared_q]

        multiplied_shares = ((first_client_share * second_client_share) + sum(qs) ) % self.__p # f(1)g(1) + q1(1) + q2(1) + ...

        for i in range(self.__n):
            self.__r[i] = (multiplied_shares * self.__A[self.__id - 1][i]) % self.__p

    def _set_r(self, party_id, shared_r):
        if self.__shared_r[party_id - 1] is not None:
            raise ValueError("r already set.")

        self.__shared_r[party_id - 1] = shared_r

    def send_r(self):
        for i in range(self.__n):
            if i == self.__id - 1:
                self.__shared_r[i] = self.__r[i]
                continue

            self.__parties[i]._set_r(self.__id, self.__r[i])

    def calculate_multiplicative_share(self):
        if self.__multiplicative_share is not None:
            raise ValueError("Coefficient already calculated.")

        self.__multiplicative_share = (
            sum([self.__shared_r[i] for i in range(self.__n)]) % self.__p
        )

    def get_multiplicative_share(self):
        return self.__multiplicative_share

    def reset(self):
        self.__q = None
        self.__shared_q = [None] * self.__n
        self.__r = None
        self.__shared_r = [None] * self.__n
        self.__multiplicative_share = None


def f(x, coefficients, p, t):
    return sum([coefficients[i] * x**i for i in range(t)]) % p


def Shamir(t, n, k0):
    p = 23
    coefficients = [random.randint(0, p - 1) for _ in range(t)]
    coefficients[0] = k0

    if coefficients[-1] == 0:
        coefficients[-1] = random.randint(1, p - 1)

    shares = []

    for i in range(1, n + 1):
        shares.append((i, f(i, coefficients, p, t)))

    return shares, p


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



def main():
    # Shamir's secret sharing
    t = 2
    n = 5
    first_secret = 3
    second_secret = 4
    third_secret = 5
    first_shares, p = Shamir(t, n, first_secret)  # First client
    second_shares, _ = Shamir(t, n, second_secret)  # Second client
    third_shares, _ = Shamir(t, n, third_secret)  # Third client

    print("shares_1 = ", first_shares)
    print("shares_2 = ", second_shares)
    print("p = ", p)

    # Create parties and set shares (P_0, ..., P_n-1)
    parties = []

    for i in range(n):
        party = Party(t, n, i + 1, p)

        parties.append(party)

    # Set the shares for each party
    for i in range(n):
        party = parties[i]

        party.set_shares(1, first_shares[i][1])
        party.set_shares(2, second_shares[i][1])
        party.set_shares(3, third_shares[i][1])

    # Set the parties for each party
    for i in range(n):
        party = parties[i]

        party.set_parties(parties)

    # Calulate A for each party
    for i in range(n):
        party = parties[i]

        party.calculate_A()

    # Calulate q for each party
    for i in range(n):
        party = parties[i]

        party.calculate_q()

    # Send q to each party
    for i in range(n):
        party = parties[i]

        party.send_q()

    # Calulate r for each party
    for i in range(n):
        party = parties[i]

        party.calculate_r(1, 2)

    # Send r to each party
    for i in range(n):
        party = parties[i]

        party.send_r()

    # Calculate the multiplicative share for each party
    for i in range(n):
        party = parties[i]

        party.calculate_multiplicative_share()

    # Sum up the multiplicative shares
    multiplicative_shares = [(0, 0)] * n

    for i in range(n):
        party = parties[i]

        multiplicative_shares[i] = (i + 1, party.get_multiplicative_share())

    print("Selected Shares for Reconstruction:")
    selected_shares = [multiplicative_shares[3], multiplicative_shares[1]]
    print(selected_shares)

    coefficients = computate_coefficients(selected_shares, p)

    print("coefficients = ", coefficients)

    secret = reconstruct_secret(selected_shares, coefficients, p)

    print("secret = ", secret % p)

    assert (first_secret * second_secret) % p == round(secret % p)

    # Reset the parties
    for i in range(n):
        party = parties[i]

        party.reset()

    #
    # New multiplication
    #

    # Calulate q for each party
    for i in range(n):
        party = parties[i]

        party.calculate_q()

    # Send q to each party
    for i in range(n):
        party = parties[i]

        party.send_q()

    # Calulate r for each party
    for i in range(n):
        party = parties[i]

        party.calculate_r(2, 3)

    # Send r to each party
    for i in range(n):
        party = parties[i]

        party.send_r()

    # Calculate the multiplicative share for each party
    for i in range(n):
        party = parties[i]

        party.calculate_multiplicative_share()

    # Sum up the multiplicative shares
    multiplicative_shares = [(0, 0)] * n

    for i in range(n):
        party = parties[i]

        multiplicative_shares[i] = (i + 1, party.get_multiplicative_share())

    print("Selected Shares for Reconstruction:")
    selected_shares = [multiplicative_shares[2], multiplicative_shares[4]]
    print(selected_shares)

    coefficients = computate_coefficients(selected_shares, p)

    print("coefficients = ", coefficients)

    secret = reconstruct_secret(selected_shares, coefficients, p)

    print("secret = ", secret % p)


if __name__ == "__main__":
    main()
