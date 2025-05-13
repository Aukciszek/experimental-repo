import copy
import os
import random

# random.seed(2137)
# print("Seed test: ",random.randint(0,100))

binary_internal = lambda n: n > 0 and [n & 1] + binary_internal(n >> 1) or []


def binary(n):
    if n == 0:
        return [0]
    else:
        return binary_internal(n)

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
        self.__shares = {} # slownik
        self.__p = p
        self.__parties = None
        self.__A = None
        self.__q = None # to co sam losuje
        self.__shared_q = [None] * n    # to co dostaje od innych
        self.__r = None
        self.__shared_r = [None] * n
        self.__multiplicative_share = None
        self.__additive_share = None
        self.__xor_share = None

        self.__random_number_bit_shares = []
        self.__random_number_share = None
        self.__comparison_a = None
        self.__z_table = None
        self.__Z_table = None
        self.__comparison_a_bits = []
        self.__res = None

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
        self.__q = Shamir(2*self.__t, self.__n, k0=0,p=self.__p)

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

        multiplied_shares = ((first_share * second_share) + sum(qs) ) % self.__p # f(1)g(1) + q1(1) + q2(1) + ...

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

        self.__additive_share = first_share + second_share

    def get_additive_share(self):
        return self.__additive_share
    
    def calculate_xor_share(self):
        self.__xor_share = self.__additive_share - 2*self.__multiplicative_share

    def get_xor_share(self):
        return self.__xor_share
    
    def set_share_to_additive_share(self, share_name:str):
        self.__shares[share_name] = self.__additive_share
    def set_share_to_multiplicative_share(self, share_name:str):
        self.__shares[share_name] = self.__multiplicative_share
    def set_share_to_xor_share(self, share_name:str):
        self.__shares[share_name] = self.__xor_share
    
    def share_exists(self,share_name:str):
        res = None
        try:
            if self.__shares[share_name] is not None:
                res = True
            else:
                res = False
        except KeyError as e:
            res = False
        return res
    
    def get_share_by_name(self,share_name:str):
        res=None
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

    def prepare_for_next_romb(self,index):
        if(index == 0):
            # prepare for last romb
            self.set_shares("x",self.__z_table[index])
            self.set_shares("X",self.__Z_table[index])
            self.set_shares("y",0)
            self.set_shares("Y",0)
        else:
            self.set_shares("x",self.__z_table[index])
            self.set_shares("X",self.__Z_table[index])
            self.set_shares("y",self.__z_table[index-1])
            self.set_shares("Y",self.__Z_table[index-1])

    # jesli party jest n-te: shares = [(n,n-ty share 1 bitu), (n,n-ty share 2 bitu), ..., (n,n-ty share ostatniego bitu)]
    def set_random_number_bit_shares(self, shares):
        for s in shares:
            self.__random_number_bit_shares.append(s)
            # [(n, n-ty share 1 bitu), (n, n-ty share 2 bitu), ...]

    def calculate_share_of_random_number(self):
        def multiply_bit_shares_by_powers_of_2(shares):
            multiplied_shares = []
            for i in range(len(shares)):
                multiplied_shares.append((shares[i][0],2**i * shares[i][1]))
            return multiplied_shares

        def add_multiplied_shares(multiplied_shares):
            party_id = multiplied_shares[0][0]
            value_of_share_r = multiplied_shares[0][1]
            for i in range(1, len(multiplied_shares)):
                value_of_share_r += multiplied_shares[i][1]
            return (party_id,value_of_share_r)

        if self.__random_number_share is not None:
            raise ValueError("Share of random number already calculated.")

        pom = multiply_bit_shares_by_powers_of_2(self.__random_number_bit_shares)
        share_of_random_number = add_multiplied_shares(pom)

        self.__random_number_share = share_of_random_number
        print("share of random number",share_of_random_number)

    # podziel_miedzy_party_losowa_liczbe_o_dlugosci_100_bitow() -> 100 * podziel_miedzy_party_losowy_bit

    def calculate_comparison_a(self, l, k, first_share_name, second_share_name):
        first_share = self.__shares[first_share_name]
        second_share = self.__shares[second_share_name]

        self.__comparison_a = (
            pow(2, l + k+1)
            - self.__random_number_share[1]
            + pow(2, l)
            + first_share
            - second_share
        )

        #print(self.__comparison_a)

    def get_comparison_a(self):
        return self.__comparison_a

    def prepare_z(self, opened_a, l, k):
        a_bin = binary(opened_a)

        while len(a_bin) < l + k :
            a_bin.append(0)

        print("a in binary: ", a_bin)

        self.__comparison_a_bits = a_bin

        self.__z_table = [None for _ in range(l)]
        self.__Z_table = [None for _ in range(l)]

        for i in range(l-1,-1,-1):
            self.__z_table[i] = self.__comparison_a_bits[i] + self.__random_number_bit_shares[i][1]
            self.__Z_table[i] = self.__comparison_a_bits[i]
        
        print("z_table",self.__z_table)
        print("Z_table",self.__Z_table)
    

    def add_comparison_a_bit_to_random_number_bit_share_and_save_as_res(self,comparison_a_bit_index,random_number_bit_share_index):
        self.__res = self.__comparison_a_bits[comparison_a_bit_index] + self.__random_number_bit_shares[random_number_bit_share_index][1]

    def add_res_to_Z(self):
        self.__res = self.__res + self.__shares["Z"]
    
    def print_test_1(self):
        print("id",self.__id,"z",self.get_share_by_name("z"),"Z",self.get_share_by_name("Z"))

    def get_res(self):
        return self.__res

    def get_random_number_share(self):
        return self.__random_number_share
    
    def get_random_number_bit_share(self,index):
        return self.__random_number_bit_shares[index]

def comparison(parties: list, opened_a:int,l: int,k:int):
    # Calulate A for each party
    for party in parties:
        party.prepare_z(opened_a, l, k)
        party.calculate_A()
    # romb the bits
    for i in range(l-1,-1,-1):
        # set x,y,X,Y
        for party in parties:
            party.prepare_for_next_romb(i)
        ### x AND y
        multiply_shares(parties,"x","y","z")
        reset_parties(parties)
        ### X XOR Y
        # Add shares
        add_shares(parties,"X","Y","Z")
        ### x AND (X XOR Y)
        multiply_shares(parties,"x","Z","Z")
        reset_parties(parties)
        ### x AND (X XOR Y) XOR X
        # Add shares
        add_shares(parties,"Z","X","Z")
        for party in parties:
            party.print_test_1()
        reset_parties(parties)
    # calculate result
    for party in parties:
        party.add_comparison_a_bit_to_random_number_bit_share_and_save_as_res(l,l)
        party.add_res_to_Z()
    # shares of the result are now stored in party.__res


def f(x, coefficients, p, t):
    return sum([coefficients[i] * x**i for i in range(t)]) % p


def Shamir(t, n, k0, p):

    coefficients = [random.randint(0, p - 1) for _ in range(t)]
    coefficients[0] = k0

    if coefficients[-1] == 0 and len(coefficients)>1:
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

# x XOR y = (x+y) - 2*(x*y)
def calculate_XOR(parties, first_share_name:str, second_share_name:str,result_share_name:str):
        # 1. self.__additive_share = x + y
        for party in parties:
            party.calculate_additive_share(first_share_name,second_share_name)
        # 2. self.__multiplicative_share = x * y
        for party in parties:
            party.calculate_q()
        for party in parties:
            party.send_q()
        for party in parties:
            party.calculate_r(first_share_name,second_share_name)
        for party in parties:
            party.send_r()
        for party in parties:
            party.calculate_multiplicative_share()
        # 3. self.__xor_share = self.__additive_share - 2 * self._multiplicative_share
        for party in parties:
            party.calculate_xor_share()
            # self.__shares[result_share_name] = self.__xor_share
            if not party.share_exists(result_share_name):
                party.set_shares(result_share_name,None)
            party.set_share_to_xor_share(result_share_name)

def multiply_shares(parties, first_share_name:str, second_share_name:str,result_share_name:str):
    # self.__multiplicative_share = x * y
    for party in parties:
        party.calculate_q()
    for party in parties:
        party.send_q()
    for party in parties:
        party.calculate_r(first_share_name,second_share_name)
    for party in parties:
        party.send_r()
    for party in parties:
        party.calculate_multiplicative_share()
        # self.__shares[result_share_name] = self.__multiplicative_share
        if not party.share_exists(result_share_name):
            party.set_shares(result_share_name,None)
        party.set_share_to_multiplicative_share(result_share_name)

def add_shares(parties, first_share_name:str, second_share_name:str,result_share_name:str):
    # self.__additive_share = x + y
    for party in parties:
        party.calculate_additive_share(first_share_name,second_share_name)
        # self.__shares[result_share_name] = self.__additive_share
        if not party.share_exists(result_share_name):
            party.set_shares(result_share_name,None)
        party.set_share_to_additive_share(result_share_name)

def reset_parties(parties):
    for party in parties:
        party.reset()

def main():
    # liczba bitów s,d = 3
    s = 4
    d = 6
    # liczba bitów d,s <= l
    l = 3
    # liczba pierwsza liczba bitów p = 4
    p = 13
    # liczba bitów p = l + k
    k = 1
    # liczba serwerow
    n = 3
    # serwery odzyskujace sekret
    t = 1

    #shares_s = Shamir(t, n, s, p)
    shares_s = [(1,4),(2,4),(3,4)]
    print("s:",s, "| shares_s:", shares_s)
    #shares_d = Shamir(t, n, d, p)
    shares_d = [(1,6),(2,6),(3,6)]
    print("d:",d, f"| shares_d: {shares_d}")
    
    # r dlugosci l+k+2 bitow
    # od najnizszej potegi
    bits_of_r = [1,0,0,0,0,1]
    oryginalne_r = sum([bits_of_r[i] * pow(2, i) for i in range(l + k + 2)])
    print(f"r: {oryginalne_r} | bits of r: {bits_of_r}")

    shares_of_bits_of_r = []
    for i in range(l + k + 2):
        new_r_bit = bits_of_r[i]
        shares_new_r_bit = Shamir(t, n, new_r_bit, p)
        #print(shares_new_r_bit)
        shares_of_bits_of_r.append(shares_new_r_bit)

    # shares for clients
    # [[share_bitu_1_dla_party_1, share_bitu_2_dla_party_1, ...], [share_bitu_2_dla_party_1, share_bitu_2_dla_party_2, ...]]
    shares_for_clients = [[] for _ in range(n)]
    for shares_of_bit in shares_of_bits_of_r:
        for i in range(n):
            shares_for_clients[i].append(shares_of_bit[i])
    print(f"shares_for_clients {shares_for_clients}")

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
        party.set_shares("s_share", shares_s[i][1])
        party.set_shares("d_share", shares_d[i][1])

    for i in range(n):
        party = parties[i]
        party.set_random_number_bit_shares(shares_for_clients[i])

    for party in parties:
        party.calculate_share_of_random_number()

    for i in range(n):
        party = parties[i]
        party.calculate_comparison_a(l, k, "s_share", "d_share")

    a_comparison_share = [(0, 0)] * n
    for i in range(n):
        party = parties[i]
        a_comparison_share[i] = (i + 1, party.get_comparison_a())

    coefficients = computate_coefficients(a_comparison_share, p)
    opened_a = reconstruct_secret(a_comparison_share, coefficients, p)
    print("opened_a = ", opened_a)

    comparison(parties,opened_a,l,k)
    
    selected_shares = [(i, parties[i].get_res()) for i in range(n)]
    coefficients = computate_coefficients(selected_shares, p)
    result = reconstruct_secret(selected_shares, coefficients, p)

    print("wynik porównania:", result)

    # reconstruct r_l
    YYY=[]
    for yyy in range(l+k+2):
        selected_shares = [(parties[i].get_random_number_bit_share(yyy)) for i in range(n)]
        #print(selected_shares)
        coefficients = computate_coefficients(selected_shares, p)
        result = reconstruct_secret(selected_shares, coefficients, p)
        YYY.append(result)
    print(f"bity r zrekonstruowane z bit sharow: {YYY}")

    # reconstruct r_l z r
    selected_shares = [parties[i].get_random_number_share() for i in range(n)]
    print("shary do rekonstrukcji r",selected_shares)
    coefficients = computate_coefficients(selected_shares, p)
    result = reconstruct_secret(selected_shares, coefficients, p)
    print("r zrekonstruowane z sharu:",result,"| binarnie:", binary(result))
    print("oryginalne r:",oryginalne_r, "| oryginalne r modulo p:",oryginalne_r%p) 




if __name__ == "__main__":
    main()
