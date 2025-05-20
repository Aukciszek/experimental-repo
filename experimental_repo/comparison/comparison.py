from party import *


def add_shares(parties, first_share_name: str, second_share_name: str, result_share_name: str):
    # self.__additive_share = x + y
    for party in parties:
        party.calculate_additive_share(first_share_name, second_share_name)
        # self.__shares[result_share_name] = self.__additive_share
        if not party.share_exists(result_share_name):
            party.set_shares(result_share_name, None)
        party.set_share_to_additive_share(result_share_name)


def multiply_shares(parties, first_share_name: str, second_share_name: str, result_share_name: str):
    # self.__multiplicative_share = x * y
    for party in parties:
        party.calculate_q()
    for party in parties:
        party.send_q()
    for party in parties:
        party.calculate_r(first_share_name, second_share_name)
    for party in parties:
        party.send_r()
    for party in parties:
        party.calculate_multiplicative_share()
        # self.__shares[result_share_name] = self.__multiplicative_share
        if not party.share_exists(result_share_name):
            party.set_shares(result_share_name, None)
        party.set_share_to_multiplicative_share(result_share_name)


# x XOR y = (x+y) - 2*(x*y)
def xor_shares(parties, first_share_name: str, second_share_name: str, result_share_name: str):
    # 1. self.__additive_share = x + y
    for party in parties:
        party.calculate_additive_share(first_share_name, second_share_name)
    # 2. self.__multiplicative_share = x * y
    for party in parties:
        party.calculate_q()
    for party in parties:
        party.send_q()
    for party in parties:
        party.calculate_r(first_share_name, second_share_name)
    for party in parties:
        party.send_r()
    for party in parties:
        party.calculate_multiplicative_share()
    # 3. self.__xor_share = self.__additive_share - 2 * self._multiplicative_share
    for party in parties:
        party.calculate_xor_share()
        # self.__shares[result_share_name] = self.__xor_share
        if not party.share_exists(result_share_name):
            party.set_shares(result_share_name, None)
        party.set_share_to_xor_share(result_share_name)


# obliczenie górnego rzędu w początkowej postaci wyrażenia: ([z]/[Z]) = ([a_l xor r_l]/[a_l]) romb ([a_l-1 xor r_l-1]/[a_l-1]) romb ... romb (0/0)
def calculate_z_tables(parties, l):
    for i in range(l - 1, -1, -1):
        calculate_z_table_XOR(parties, i)
        reset_parties(parties)


# obliczenie [a_i xor r_i]
def calculate_z_table_XOR(parties, index: int):
    # 1. self.__additive_share = x + y
    for party in parties:
        party.calculate_additive_share_of_z_table_arguments(index)
    # 2. self.__multiplicative_share = x * y
    for party in parties:
        party.calculate_q()
    for party in parties:
        party.send_q()
    for party in parties:
        party.calculate_r_of_z_table_arguments(index)
    for party in parties:
        party.send_r()
    for party in parties:
        party.calculate_multiplicative_share()
    # 3. self.__xor_share = self.__additive_share - 2 * self._multiplicative_share
    for party in parties:
        party.calculate_xor_share()
        party.set_z_table_to_xor_share(index)


def reset_parties(parties):
    for party in parties:
        party.reset()


# Compare s,d and save the resulting shares in share named "res"
# d >= s  -->  res=1
# d < s   -->  res=0
def comparison(parties: list, opened_a: int, l: int, k: int):
    # Prepare z and Z initial values
    for party in parties:
        party.prepare_z_tables(opened_a, l, k)
    for i in range(l):
        calculate_z_tables(parties, l)
    for party in parties:
        party.initialize_z_and_Z(l)
    print("po xorze gornego rzedu")
    parties[0].print_test_z_tables()
    # Romb the bits
    for i in range(l - 1, -1, -1):
        # set x,y,X,Y
        for party in parties:
            party.prepare_for_next_romb(i)
            party.print_test_2()
        ### x AND y
        multiply_shares(parties, "x", "y", "z")
        reset_parties(parties)
        print("x AND y", end="\t" * 5)
        parties[0].print_test_1()
        ### X XOR Y
        xor_shares(parties, "X", "Y", "Z")
        reset_parties(parties)
        # print("X XOR Y",end="\t"*5)
        # parties[0].print_test_1()
        ### x AND (X XOR Y)
        multiply_shares(parties, "x", "Z", "Z")
        reset_parties(parties)
        # print("x AND (X XOR Y)",end="\t"*3)
        # parties[0].print_test_1()
        ### x AND (X XOR Y) XOR X
        xor_shares(parties, "Z", "X", "Z")
        print("x AND (X XOR Y) XOR X", end="\t" * 1)
        for party in parties:
            party.print_test_1()
        reset_parties(parties)
    # calculate result
    # [res] = a_l XOR [r_l] XOR [Z]
    for party in parties:
        party.prepare_shares_for_res_xors(l, l)

    selected_shares = [(1, parties[0].get_share_by_name("a_l")), (4, parties[3].get_share_by_name("a_l"))]
    print(selected_shares)
    coefficients = computate_coefficients(selected_shares, 1013)
    opened = reconstruct_secret(selected_shares, coefficients, 1013)
    print("opened a_l = ", opened)
    selected_shares = [(1, parties[0].get_share_by_name("r_l")), (4, parties[3].get_share_by_name("a_l"))]
    print(selected_shares)
    coefficients = computate_coefficients(selected_shares, 1013)
    opened = reconstruct_secret(selected_shares, coefficients, 1013)
    print("opened r_l = ", opened)

    xor_shares(parties, "a_l", "r_l", "res")
    reset_parties(parties)
    xor_shares(parties, "res", "Z", "res")
    print("a_l XOR r_l XOR Z", end="\t" * 2)
    parties[0].print_test_3()
    reset_parties(parties)
    # shares of the result are now stored in share named "res"


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

    # shares_s = Shamir(t, n, s, p)
    shares_s = [(1, 4), (2, 4), (3, 4)]
    print("s:", s, "| shares_s:", shares_s)
    # shares_d = Shamir(t, n, d, p)
    shares_d = [(1, 6), (2, 6), (3, 6)]
    print("d:", d, f"| shares_d: {shares_d}")

    # r dlugosci l+k+2 bitow
    # od najnizszej potegi
    bits_of_r = [1, 0, 0, 0, 0, 1]
    oryginalne_r = sum([bits_of_r[i] * pow(2, i) for i in range(l + k + 2)])
    print(f"r: {oryginalne_r} | bits of r: {bits_of_r}")

    shares_of_bits_of_r = []
    for i in range(l + k + 2):
        new_r_bit = bits_of_r[i]
        shares_new_r_bit = Shamir(t, n, new_r_bit, p)
        # print(shares_new_r_bit)
        shares_of_bits_of_r.append(shares_new_r_bit)

    # shares for clients
    # [[share_bitu_1_dla_party_1, share_bitu_2_dla_party_1, ...], [share_bitu_2_dla_party_1, share_bitu_2_dla_party_2, ...]]
    shares_for_clients = [[] for _ in range(n)]
    for shares_of_bit in shares_of_bits_of_r:
        for i in range(n):
            shares_for_clients[i].append(shares_of_bit[i])
    print(f"shares_for_clients {shares_for_clients}")

    # Create parties
    parties = []
    for i in range(n):
        party = Party(t, n, i + 1, p)
        parties.append(party)

    # Set the parties for each party
    for i in range(n):
        party = parties[i]
        party.set_parties(parties)

    # Calulate A for each party
    for party in parties:
        party.calculate_A()

    # Set the shares for each party
    for i in range(n):
        party = parties[i]
        party.set_shares("s_share", shares_s[i][1])
        party.set_shares("d_share", shares_d[i][1])
        party.set_random_number_bit_shares(shares_for_clients[i])
        party.calculate_share_of_random_number()

    # Calculate comparison a
    # [a] = 2^(l+k+1) - [r] + 2^l + [d] - [s]
    # first share   --> d_share
    # second share  --> s_share
    for i in range(n):
        party = parties[i]
        party.calculate_comparison_a(l, k, "d_share", "s_share")

    # Open comparison a
    a_comparison_share = [(0, 0)] * n
    for i in range(n):
        party = parties[i]
        a_comparison_share[i] = (i + 1, party.get_comparison_a())

    # Select shares for reconstruction
    indeksy = [_ for _ in range(n)]
    wybrane_indeksy = []
    for _ in range(t):
        w = random.choice(indeksy)
        wybrane_indeksy.append(w)
        indeksy.remove(w)
    selected_shares = [a_comparison_share[_] for _ in wybrane_indeksy]
    # print(selected_shares)
    coefficients = computate_coefficients(selected_shares, p)
    opened_a = reconstruct_secret(selected_shares, coefficients, p)
    print("opened_a = ", opened_a)

    # Compare s,d and save the resulting shares in share named "res"
    # d >= s  -->  res=1
    # d < s   -->  res=0
    comparison(parties, opened_a, l, k)

    # Select shares for reconstruction
    indeksy = [_ for _ in range(n)]
    wybrane_indeksy = []
    for _ in range(t):
        w = random.choice(indeksy)
        wybrane_indeksy.append(w)
        indeksy.remove(w)
    selected_shares = [(i, parties[i].get_share_by_name("res")) for i in wybrane_indeksy]
    # print(selected_shares)
    # Reconstruct res
    coefficients = computate_coefficients(selected_shares, p)
    result = reconstruct_secret(selected_shares, coefficients, p)

    print("wynik porównania:", result)


if __name__ == "__main__":
    main()
