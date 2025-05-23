from sharing_utils import *
from arithmetic_utils import *


class Party:
    def __init__(self, t, n, id, p):
        # parametry
        self.__t = t
        self.__n = n
        self.__p = p
        # dane strony
        self.__id = id
        self.__parties = None
        # udzialy
        self.__shares = {}  # slownik
        # zmienne do operacji na udzialach
        self.__A = None
        self.__q = None  # to co sam losuje
        self.__shared_q = [None] * n  # to co dostaje od innych
        self.__r = None
        self.__shared_r = [None] * n
        # wyniki operacji na udzialach
        self.__multiplicative_share = None
        self.__additive_share = None
        self.__xor_share = None
        # zmienne do operacji porownania
        self.__random_number_bit_shares = []
        self.__random_number_share = None
        self.__comparison_a = None
        self.__z_table = None
        self.__Z_table = None
        self.__comparison_a_bits = []
        # zmienne do random bit
        self.__part_of_u = None  # to co sam losuje
        self.__shared_part_of_u = [None] * n  # to co dostaje od innych

    def get_id(self):
        return self.__id

    def set_shares(self, share_name: str, share):
        self.__shares[share_name] = share

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
        self.__q = Shamir(2 * self.__t, self.__n, k0=0, p=self.__p)

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

    def calculate_r(self, first_share_name: str, second_share_name: str):
        if self.__r is not None:
            raise ValueError("r already calculated.")

        self.__r = [0] * self.__n

        first_share = self.__shares[first_share_name]
        second_share = self.__shares[second_share_name]

        # receive q from other parties
        # add sum of qs in multiplied shares
        qs = [x[1] for x in self.__shared_q]

        multiplied_shares = ((first_share * second_share) + sum(qs)) % self.__p  # f(1)g(1) + q1(1) + q2(1) + ...

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

    def calculate_additive_share(self, first_share_name: str, second_share_name: str):
        if self.__additive_share is not None:
            raise ValueError("Coefficient already calculated.")

        first_share = self.__shares[first_share_name]
        second_share = self.__shares[second_share_name]

        self.__additive_share = (first_share + second_share) % self.__p

    def get_additive_share(self):
        return self.__additive_share

    def calculate_xor_share(self):
        self.__xor_share = (self.__additive_share - 2 * self.__multiplicative_share) % self.__p

    def get_xor_share(self):
        return self.__xor_share

    def set_share_to_additive_share(self, share_name: str):
        self.__shares[share_name] = self.__additive_share

    def set_share_to_multiplicative_share(self, share_name: str):
        self.__shares[share_name] = self.__multiplicative_share

    def set_share_to_xor_share(self, share_name: str):
        self.__shares[share_name] = self.__xor_share

    def share_exists(self, share_name: str):
        res = None
        try:
            if self.__shares[share_name] is not None:
                res = True
            else:
                res = False
        except KeyError as e:
            res = False
        return res

    def get_share_by_name(self, share_name: str):
        res = None
        if self.share_exists(share_name):
            res = self.__shares[share_name]
        return res

    def reset(self):
        self.__q = None
        self.__shared_q = [None] * self.__n
        self.__r = None
        self.__shared_r = [None] * self.__n
        self.__multiplicative_share = None
        self.__additive_share = None
        self.__xor_share = None

    ### funkcje do porownania

    def set_random_number_bit_shares(self, shares):
        for s in shares:
            self.__random_number_bit_shares.append(s)
            # [(n, n-ty share 1 bitu), (n, n-ty share 2 bitu), ...]

    def set_random_number_bit_share_to_temporary_random_bit_share(self, bit_index):
        while len(self.__random_number_bit_shares) < bit_index + 1:
            self.__random_number_bit_shares.append(None)
        self.__random_number_bit_shares[bit_index] = (self.__id, self.get_share_by_name("temporary_random_bit"))

    def get_random_number_bit_share(self, index):
        return self.__random_number_bit_shares[index]

    def calculate_share_of_random_number(self):
        def multiply_bit_shares_by_powers_of_2(shares):
            multiplied_shares = []
            for i in range(len(shares)):
                multiplied_shares.append((shares[i][0], 2 ** i * shares[i][1]))
            return multiplied_shares

        def add_multiplied_shares(multiplied_shares):
            party_id = multiplied_shares[0][0]
            value_of_share_r = multiplied_shares[0][1]
            for i in range(1, len(multiplied_shares)):
                value_of_share_r += multiplied_shares[i][1]
            return (party_id, value_of_share_r % self.__p)  # r musi byc mniejsze niz p inaczej nie wychodzi

        if self.__random_number_share is not None:
            raise ValueError("Share of random number already calculated.")

        pom = multiply_bit_shares_by_powers_of_2(self.__random_number_bit_shares)
        share_of_random_number = add_multiplied_shares(pom)

        self.__random_number_share = share_of_random_number

    def get_random_number_share(self):
        return self.__random_number_share

    # Calculate comparison a
    # [a] = 2^(l+k+1) - [r] + 2^l + [d] - [s]
    # first share   --> d_share
    # second share  --> s_share
    def calculate_comparison_a(self, l, k, first_share_name, second_share_name):
        first_share = self.__shares[first_share_name]
        second_share = self.__shares[second_share_name]
        self.__comparison_a = (
                pow(2, l + k + 1)
                - self.__random_number_share[1]
                + pow(2, l)
                + first_share
                - second_share
        )  # nie modulowac !

    def get_comparison_a(self):
        return self.__comparison_a

    def prepare_z_tables(self, opened_a, l, k):
        a_bin = binary(opened_a)

        while len(a_bin) < l + k:
            a_bin.append(0)
        self.__comparison_a_bits = a_bin

        self.__z_table = [None for _ in range(l)]
        self.__Z_table = [None for _ in range(l)]

        for i in range(l - 1, -1, -1):
            self.__z_table[i] = self.__comparison_a_bits[i]  # pozniej XOR z self.__random_number_bit_shares[i][1]
            self.__Z_table[i] = self.__comparison_a_bits[i]

    def initialize_z_and_Z(self, l):
        self.set_shares("z", self.__z_table[l - 1])
        self.set_shares("Z", self.__Z_table[l - 1])

    # najpierw zZ[l-1] ◇ zZ[l-2], potem wynik ◇ zZ[l-3], ... , wynik ◇ zZ[0]
    # na koniec dodatkowo wynik ◇ (0|0)
    def prepare_for_next_romb(self, index):
        self.set_shares("x", self.get_share_by_name("z"))
        self.set_shares("X", self.get_share_by_name("Z"))
        if index == 0:
            # prepare for last romb
            self.set_shares("y", 0)
            self.set_shares("Y", 0)
        else:
            self.set_shares("y", self.__z_table[index - 1])
            self.set_shares("Y", self.__Z_table[index - 1])

    ### do operacji arytmetycznych w ramach rombu

    # zwykle dodawanie ale bitow a z sherami bitow r
    def calculate_additive_share_of_z_table_arguments(self, index):
        first_share = self.__comparison_a_bits[index]
        second_share = self.__random_number_bit_shares[index][1]

        self.__additive_share = (first_share + second_share) % self.__p

    # zwykle mnozenie ale bitow a z sherami bitow r
    def calculate_r_of_z_table_arguments(self, index):
        if self.__r is not None:
            raise ValueError("r already calculated.")

        self.__r = [0] * self.__n

        first_share = self.__comparison_a_bits[index]
        second_share = self.__random_number_bit_shares[index][1]

        # receive q from other parties
        # add sum of qs in multiplied shares
        qs = [x[1] for x in self.__shared_q]

        multiplied_shares = ((first_share * second_share) + sum(qs)) % self.__p  # f(1)g(1) + q1(1) + q2(1) + ...

        for i in range(self.__n):
            self.__r[i] = (multiplied_shares * self.__A[self.__id - 1][i]) % self.__p

    def set_z_table_to_xor_share(self, index):
        self.__z_table[index] = self.__xor_share

    ### do obliczenia wyniku porownania
    # [res] = a_l XOR [r_l] XOR [Z]

    def prepare_shares_for_res_xors(self, comparison_a_bit_index, random_number_bit_share_index):
        self.set_shares("a_l", self.__comparison_a_bits[comparison_a_bit_index])
        self.set_shares("r_l", self.__random_number_bit_shares[random_number_bit_share_index][1])

    ### testowe printy

    def print_test_1(self):
        print(f"udzialy party {self.__id}:", "z", self.__id, self.get_share_by_name("z"), "Z",
              self.get_share_by_name("Z"))

    def print_test_2(self):
        print(f"udzialy party {self.__id}:", "x", self.get_share_by_name("x"), "y", self.get_share_by_name("y"), "X",
              self.get_share_by_name("X"), "Y",
              self.get_share_by_name("Y"))

    def print_test_3(self):
        print(f"udzialy party {self.__id}:", "a_l", self.get_share_by_name("a_l"), "r_l", self.get_share_by_name("r_l"),
              "Z",
              self.get_share_by_name("Z"), "res", self.get_share_by_name("res"))

    def print_test_z_tables(self):
        print(f"udzialy party {self.__id}: z {self.__z_table} Z {self.__Z_table}")

    ### random bit

    def calculate_part_of_u(self):
        self.__part_of_u = Shamir(self.__t, self.__n, random.randint(1, self.__p), self.__p)

    # set part_of_u to other parties
    def _set_part_of_u(self, party_id, shared_part_of_u):
        self.__shared_part_of_u[party_id - 1] = shared_part_of_u

    # send part_of_u to other parties
    def send_part_of_u(self):
        for i in range(self.__n):
            if i == self.__id - 1:
                self.__shared_part_of_u[i] = self.__part_of_u[i]
                continue
            self.__parties[i]._set_part_of_u(self.__id, self.__part_of_u[i])

    def calculate_u(self):
        # receive part_of_u from other parties
        part_of_u_s = [x[1] for x in self.__shared_part_of_u]
        # save sum of part_of_u_s as a share of u
        self.set_shares("u", sum(part_of_u_s) % self.__p)
