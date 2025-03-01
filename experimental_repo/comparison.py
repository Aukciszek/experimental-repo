import os
import random

binary_internal = lambda n: n > 0 and [n & 1] + binary_internal(n >> 1) or []



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
        self.__random_number_bit_shares = []
        self.__random_number_share = None
        self.__p = p
        self.__parties = None
        self.__A = None
        self.__r = None
        self.__shared_r = [None] * n
        self.__multiplicative_share = None
        self.__additive_share = None
        self.__multiplication_results = []

    def set_shares(self, client_id, share):
        self.__client_shares.append((client_id, share))
    
    # jesli party jest n-te: [n-ty share 1 bitu, n-ty share 2 bitu, ..., n-ty share ostatniego bitu]
    def set_random_number_bit_shares(self, shares):
        for i, share in enumerate(shares):
            self.__random_number_bit_shares.append((i + 1, share))

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
        multiplied_shares = (first_client_share * second_client_share) % self.__p

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
    
    def calculate_additive_share(self, first_client_id, second_client_id):
        if self.__additive_share is not None:
            raise ValueError("Coefficient already calculated.")
        
        first_client_share = next(
            (y for x, y in self.__client_shares if x == first_client_id), None
        )
        second_client_share = next(
            (y for x, y in self.__client_shares if x == second_client_id), None
        )

        self.__additive_share = first_client_share + second_client_share

    def get_additive_share(self):
        return self.__additive_share
    
    # def multiply_by_constant(self, constant, share_id, new_share_id):
    #     wyn=constant*self.__client_shares[share_id][1]
    #     self.set_shares(new_share_id,wyn)
    
    def calculate_share_of_random_number(self):
        def multiply_bit_shares_by_powers_of_2(shares):
            multiplied_shares = []
            for i in range(len(shares)):
                multiplied_shares.append(2 ** i * shares[i][1])
            return multiplied_shares
    
        def add_multiplied_shares(multiplied_shares):
            share_r=multiplied_shares[0]
            for i in range(1,len(multiplied_shares)):
                share_r+=multiplied_shares[i]
            return share_r 

        if self.__random_number_share is not None:
            raise ValueError("Share of random number already calculated.")
        
        pom = multiply_bit_shares_by_powers_of_2(self.__random_number_bit_shares)
        share_of_random_number = add_multiplied_shares(pom)

        self.__random_number_share = share_of_random_number

    # podziel_miedzy_party_losowa_liczbe_o_dlugosci_100_bitow() -> 100 * podziel_miedzy_party_losowy_bit

    def reset(self):
        self.__r = None
        self.__shared_r = [None] * self.__n
        self.__multiplicative_share = None
        self.__additive_share = None

    # FUNKCJE
    # 1. pomnóż wszystkie sharingi bitów przez potęgę 2 (2^i) DONE SKONCZONE
    # 2. oblicz r na podstawie sharingów bitów r
    # 3. 
    # 4. 
    # 5. 

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

def new_and(a,b):
    return a & b

def romb(x, X, y, Y):
    return (x & y, x & (X ^ Y) ^ X)


def main():
    s = 3
    d = 7

    # liczba bitów p = k+l
    k = 1
    # liczba bitów d,s <= l
    l = 3
    
    # liczba serwerow
    n = 5
    # serwery odzyskujace sekret
    t = 2
    # liczba pierwsza
    p=13

    shares_s = Shamir(t, n, s,p)
    print(s,"shares_s: ", shares_s)
    shares_d = Shamir(t, n, d,p)
    print(d,f'shares_d: {shares_d}')

    # sprawdzenie d
    selected_shares = [shares_d[3], shares_d[1]]
    coefficients = computate_coefficients(selected_shares, p)
    secret = reconstruct_secret(selected_shares, coefficients, p)
    
    print("reconstructed d: ", secret % p)

    bits_of_r = []
    shares_of_bits_of_r = []
    for i in range(l + k + 2):
        new_r_bit = int.from_bytes(os.urandom(1)) % 2
        bits_of_r.append(new_r_bit)
        shares_new_r_bit, p_new_r_bit = Shamir(t, n, new_r_bit,p)
        shares_of_bits_of_r.append(shares_new_r_bit)

    print("bits of r: ", bits_of_r)
    print("l-th bit of r: ", bits_of_r[l],"shares of l-th bit of r: ", shares_of_bits_of_r[l])

    #r = sum([bits_of_r[i] * pow(2, i) for i in range(l + k + 2)])
    

    # Create parties and set shares (P_0, ..., P_n-1)
    parties = []
    for i in range(n):
        party = Party(t, n, i + 1, p)
        parties.append(party)

    # Set the shares for each party
    for i in range(n):
        party = parties[i]
        party.set_shares(1, shares_s[i][1])
        party.set_shares(2, shares_d[i][1])
        for j in range(len(bits_of_r)):
            party.set_shares(j+3,shares_of_bits_of_r[j][i][1])

    # Set the parties for each party
    for i in range(n):
        party = parties[i]
        party.set_parties(parties)

    # Calulate A for each party
    for i in range(n):
        party = parties[i]
        party.calculate_A()

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

    

    a = pow(2, l + k + 1) - r + pow(2, l) + d - s
    if a < 0:
        print("a: ", a)
        print("a is negative")
        return

    print("a: ", a)

    a_bin = binary(a)
    r_prim_bin = binary(r)

    while len(a_bin) < l + k + 2:
        a_bin.append(0)

    while len(r_prim_bin) < l + k + 2:
        r_prim_bin.append(0)

    print("a in binary: ", a_bin)
    print("r_prim in binary: ", r_prim_bin)

    z = []

    for i in range(l):
        z.append((a_bin[i] ^ r_prim_bin[i], a_bin[i]))

    z = list(reversed(z))
    z.append((0, 0))

    print("z: ", z)

    while len(z) > 1:
        z[0] = romb(z[0][0], z[0][1], z[1][0], z[1][1])
        z.pop(1)

    print("z: ", z)

    res = a_bin[l] ^ r_prim_bin[l] ^ z[0][1]

    print("res: ", res)

    assert res == 1


if __name__ == "__main__":
    main()
