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
        self.__comparison_a = None
        self.__z = None
        self.__Z = None
        self.__comparison_a_bits = []
        self.__x = None
        self.__X = None
        self.__y = None
        self.__Y = None
        self.__res = None

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

    def add_client_shares(self, first_client_id, second_client_id):
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
    
    def calculate_share_of_random_number(self):
        def multiply_bit_shares_by_powers_of_2(shares):
            multiplied_shares = []
            for i in range(len(shares)):
                multiplied_shares.append(2**i * shares[i][1])
            return multiplied_shares

        def add_multiplied_shares(multiplied_shares):
            share_r = multiplied_shares[0]
            for i in range(1, len(multiplied_shares)):
                share_r += multiplied_shares[i]
            return share_r

        if self.__random_number_share is not None:
            raise ValueError("Share of random number already calculated.")

        pom = multiply_bit_shares_by_powers_of_2(self.__random_number_bit_shares)
        share_of_random_number = add_multiplied_shares(pom)

        self.__random_number_share = share_of_random_number

    # podziel_miedzy_party_losowa_liczbe_o_dlugosci_100_bitow() -> 100 * podziel_miedzy_party_losowy_bit

    def calculate_comparison_a(self, l, k, first_client_id, second_client_id):
        first_client_share = next(
            (y for x, y in self.__client_shares if x == first_client_id), None
        )
        second_client_share = next(
            (y for x, y in self.__client_shares if x == second_client_id), None
        )

        self.__comparison_a = (
            pow(2, l + k-1)
            - self.__random_number_share
            + pow(2, l)
            + first_client_share
            - second_client_share
        )

    def get_comparison_a(self):
        return self.__comparison_a

    def prepare_z(self, opened_a, l, k):
        a_bin = binary(opened_a)

        while len(a_bin) < l + k :
            a_bin.append(0)

        print("a in binary: ", a_bin)

        self.__comparison_a_bits = a_bin

    def add_comparison_a_bit_to_random_number_bit_share_and_save_as_z(self, comparison_a_bit_index, random_number_bit_share_index):        
        self.__x = self.__comparison_a_bits[comparison_a_bit_index] + self.__random_number_bit_shares[random_number_bit_share_index][1]

    def calculate_r_of_x_AND_y(self):
        self.__r = [0] * self.__n

        multiplied_shares = (self.__x * self.__y) % self.__p

        for i in range(self.__n):
            self.__r[i] = (multiplied_shares * self.__A[self.__id - 1][i]) % self.__p

    def set_z_to_multiplicative_share(self):
        self.__z = self.__multiplicative_share

    def add_X_to_Y(self):
        self.__Z = self.__X + self.__Y

    def calculate_r_of_x_AND_Z(self):
        self.__r = [0] * self.__n

        multiplied_shares = (self.__X * self.__Z) % self.__p

        for i in range(self.__n):
            self.__r[i] = (multiplied_shares * self.__A[self.__id - 1][i]) % self.__p
    
    def set_Z_to_multiplicative_share(self):
        self.__Z = self.__multiplicative_share
    
    def add_Z_to_X(self):
        self.__Z = self.__Z + self.__X

    def prepare_for_next_romb(self,index):
        # 1. self.__x = add_comparison_a_bit_to_random_number_bit_share(index)
        # 2. self.__X = self.__comparison_a_bits(index)
        # 3. self.__y = self.__z
        # 4. self.__Y = self.__Z
        if(index == 0):
            # prepare for first romb
            self.add_comparison_a_bit_to_random_number_bit_share_and_save_as_z(index,index)
            self.__X = self.__comparison_a_bits[index]
            self.__y = 0
            self.__Y = 0
        else:
            self.add_comparison_a_bit_to_random_number_bit_share_and_save_as_z(index,index)
            self.__X = self.__comparison_a_bits[index]
            self.__y = self.__z
            self.__Y = self.__Z

    def add_comparison_a_bit_to_random_number_bit_share_and_save_as_res(self,comparison_a_bit_index,random_number_bit_share_index):
        self.__res = self.__comparison_a_bits[comparison_a_bit_index] + self.__random_number_bit_shares[random_number_bit_share_index][1]

    def add_res_to_Z(self):
        self.__res = self.__res + self.__Z

    def get_res(self):
        return self.__res
    
    def reset(self):
        self.__r = None
        self.__shared_r = [None] * self.__n
        self.__multiplicative_share = None
        self.__additive_share = None

    # FUNKCJE
    # 1. pomnóż wszystkie sharingi bitów przez potęgę 2 (2^i) DONE SKONCZONE
    # 2. oblicz r na podstawie sharingów bitów r
    # 3. ???
    """
    Helper fuction for comparison
    (x, X) ◇ (y, Y) = (x^y , x^(X⊕Y)⊕X)
    x - zZ[0][0]  X - zZ[0][1]
    y - zZ[0][0]  Y - zZ[0][1]
    """

def comparison(parties: list,l: int):
    # 0. calculate_A()
    # Calulate A for each party
    for party in parties:
        party.calculate_A()

    for i in range(l):
        # 1. self.__x = add_comparison_a_bit_to_random_number_bit_share(index)
        # 2. self.__X = self.__comparison_a_bits(index)
        # 3. self.__y = self.__z
        # 4. self.__Y = self.__Z
        for party in parties:
            party.prepare_for_next_romb(i)
        # 1. calculate_r_of_x_AND_y()
        # Calulate r for each party
        for party in parties:
            party.calculate_r_of_x_AND_y()
        # 2. send_r()
        # Send r to each party
        for party in parties:
            party.send_r()
        # 3. calculate_multiplicative_share()
        # Calculate the multiplicative share for each party
        for party in parties:
            party.calculate_multiplicative_share()
        # 4. self.__z = self.__multiplicative_share
        # Set z to be the multiplicative share
        for party in parties:
            party.set_z_to_multiplicative_share()
            party.reset()
        # 5. self.__Z = add_X_to_Y()
        for party in parties:
            party.add_X_to_Y()
        # 6. calculate_r_of_x_AND_Z()
        for party in parties:
            party.calculate_r_of_x_AND_Z()
        # 7. send_r()
        for party in parties:
            party.send_r()
        # 8. calculate_multiplicative_share()
        for party in parties:
            party.calculate_multiplicative_share()
        # 9. self.__Z = self.__multiplicative_share
        for party in parties:
            party.set_Z_to_multiplicative_share()
            party.reset()
        # 10. self.__Z = add_Z_to_X()
        for party in parties:
            party.add_Z_to_X()

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


def romb1(x, X, y, Y):
    return (x & y, x & (X ^ Y) ^ X)


def romb(parties, l):
    x = [p.__zZ[0][0] for p in parties]
    X = [p.__zZ[0][1] for p in parties]
    for i in range(1, l):
        for p in parties:
            # 1 x[p] = p.calculate_multiplicative_share(x[p], p.__zZ[i][0])
            # 2 pom = p.calculate_additive_share(X[p], p.__zZ[i][1])
            # 3 pom = p.calculate_multiplicative_share(x[p], pom)
            # 4 X[p] = p.calculate_additive_share(pom, X[p])

            # 1
            p.calculate_A()
            p.calculate_r_2(x[p], p.__zZ[i][0])
            p.send_r()
            p.calculate_multiplicative_share()
            x[p] = p.get_multiplicative_share()
            # 2
            p.calculate_additive_share_2(X[p], p.__zZ[i][1])
            pom = p.get_additive_share()
            # 3
            p.calculate_A()
            p.calculate_r_2(x[p], pom)
            p.send_r()
            p.calculate_multiplicative_share()
            pom = p.get_multiplicative_share()
            # 4
            p.calculate_additive_share_2(pom, X[p])
            X[p] = p.get_additive_share()

    return x, X


# x[p]    p.__zZ[i][0]
# X[p]    p.__zZ[i][1]

def main():
    s = 3
    d = 7

    # liczba bitów p = k+l
    k = 4
    # liczba bitów d,s <= l
    l = 3

    # liczba serwerow
    n = 5
    # serwery odzyskujace sekret
    t = 2
    # liczba pierwsza
    p = 97

    shares_s = Shamir(t, n, s, p)
    print(s, "shares_s: ", shares_s)
    shares_d = Shamir(t, n, d, p)
    print(d, f"shares_d: {shares_d}")
    

    # sprawdzenie d
    selected_shares = [shares_d[3], shares_d[1]]
    coefficients = computate_coefficients(selected_shares, p)
    secret = reconstruct_secret(selected_shares, coefficients, p)

    print("reconstructed d: ", secret % p)

    # bits_of_r = []
    # shares_of_bits_of_r = []
    # for i in range(l + k -1):
    #     new_r_bit = int.from_bytes(os.urandom(1)) % 2
    #     bits_of_r.append(new_r_bit)
    #     shares_new_r_bit = Shamir(t, n, new_r_bit, p)
    #     shares_of_bits_of_r.append(shares_new_r_bit)

    bits_of_r = [1,0,0,1,1,1]
    shares_of_bits_of_r = []
    for i in range(l + k -1):
        new_r_bit = bits_of_r[i]
        shares_new_r_bit = Shamir(t, n, new_r_bit, p)
        shares_of_bits_of_r.append(shares_new_r_bit)

    # print("bits of r: ", bits_of_r)
    # print(
    #     "l-th bit of r: ",
    #     bits_of_r[l],
    #     "shares of l-th bit of r: ",
    #     shares_of_bits_of_r[l],
    # )

    # r = sum([bits_of_r[i] * pow(2, i) for i in range(l + k + 2)])

    # Create parties and set shares (P_0, ..., P_n-1)
    parties = []
    for i in range(n):
        party = Party(t, n, i + 1, p)
        parties.append(party)

    # Set the parties for each party
    for i in range(n):
        party = parties[i]
        party.set_parties(parties)

    # Set the shares for each party
    for i in range(n):
        party = parties[i]
        party.set_shares(1, shares_s[i][1])
        party.set_shares(2, shares_d[i][1])

    print(shares_of_bits_of_r)

    shares_for_clients = [[] for _ in range(n)]

    for bit in shares_of_bits_of_r:
        for i, share_of_bit in enumerate(bit):
            shares_for_clients[i].append(share_of_bit[1])

    for i in range(n):
        party = parties[i]
        party.set_random_number_bit_shares(shares_for_clients[i])

    for i in range(n):
        party = parties[i]
        party.calculate_share_of_random_number()

    for i in range(n):
        party = parties[i]
        party.calculate_comparison_a(l, k, 1, 2)

    a_comparison_share = [(0, 0)] * n

    for i in range(n):
        party = parties[i]

        a_comparison_share[i] = (i + 1, party.get_comparison_a())

    coefficients = computate_coefficients(a_comparison_share, p)
    opened_a = reconstruct_secret(a_comparison_share, coefficients, p)

    print("opened_a = ", opened_a)

    for i in range(n):
        party = parties[i]
        party.prepare_z(opened_a, l, k)

    comparison(parties,l)

    for party in parties:
        party.add_comparison_a_bit_to_random_number_bit_share_and_save_as_res(l,l)
        party.add_res_to_Z()
    
    selected_shares = [(i, parties[i].get_res()) for i in range(n)]
    coefficients = computate_coefficients(selected_shares, p)
    result = reconstruct_secret(selected_shares, coefficients, p)

    print(result)

    # while len(z) > 1:
    #     z[0] = romb(z[0][0], z[0][1], z[1][0], z[1][1])
    #     z.pop(1)

    # print("z: ", z)

    # res = a_bin[l] ^ r_prim_bin[l] ^ z[0][1]

    # print("res: ", res)

    # assert res == 1


if __name__ == "__main__":
    main()
