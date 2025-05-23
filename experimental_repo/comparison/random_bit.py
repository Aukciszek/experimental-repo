from comparison import *


def smallest_square_root_modulo(number, modulus):
    wyn = None
    for i in range(modulus):
        if (i * i) % modulus == number:
            wyn = i
            break
    return wyn


def share_random_u(parties, p, n, t):
    for party in parties:
        party.calculate_part_of_u()
    for party in parties:
        party.send_part_of_u()
    for party in parties:
        party.calculate_u()


def share_random_bit(parties, p, n, t, bit_index):
    opened_v = 0
    while opened_v <= 0:
        party: Party
        # All servers secret share a random value,
        # and add all shares locally, to form a sharing [u] of a random unknown u.
        share_random_u(parties, p, n, t)
        # We then compute [v] = [u2 mod p] and open v.
        multiply_shares(parties, "u", "u", "v")
        reset_parties(parties)
        # Select shares for reconstruction
        indeksy = [_ for _ in range(n)]
        wybrane_indeksy = []
        for _ in range(t):
            pom = random.choice(indeksy)
            wybrane_indeksy.append(pom)
            indeksy.remove(pom)
        selected_shares = [(parties[i].get_id(), parties[i].get_share_by_name("u")) for i in wybrane_indeksy]
        print("random_shares:", [(party.get_id(), party.get_share_by_name("u")) for party in parties])
        print("selected_shares:", selected_shares)
        coefficients = computate_coefficients(selected_shares, p)
        opened_w = reconstruct_secret(selected_shares, coefficients, p)
        print("opened_u_party = ", opened_w)
        # Select shares for reconstruction
        indeksy = [_ for _ in range(n)]
        wybrane_indeksy = []
        for _ in range(t):
            pom = random.choice(indeksy)
            wybrane_indeksy.append(pom)
            indeksy.remove(pom)
        selected_shares = [(i + 1, parties[i].get_share_by_name("v")) for i in wybrane_indeksy]
        # print(selected_shares)
        coefficients = computate_coefficients(selected_shares, p)
        opened_v = reconstruct_secret(selected_shares, coefficients, p)
        print("opened_v = ", opened_v)
    #  If v = 0 we start over,
    #  otherwise we publicly compute a square root w of v, say we choose the smallest one.
    w = smallest_square_root_modulo(opened_v, p)
    print("w", w)
    # We compute w^(-1)*[u] mod p
    # which will be 1 with probability 50% and 1 with probability 50%.
    inverse_w = modular_multiplicative_inverse(w, p)
    # calculate w^(-1)*[u]
    for party in parties:
        party.set_shares("dummy_sharing_of_inverse_w_", inverse_w)
    multiply_shares(parties, "dummy_sharing_of_inverse_w_", "u", "inverse_w_times_u")
    reset_parties(parties)
    # Therefore, [ ((w^(-1)*[u] + 1) * 2^(-1) ) mod p]
    # will produce the random shared binary value we wanted.
    # calculate w^(-1)*[u] + 1
    for party in parties:
        party.set_shares("dummy_sharing_of_one", 1)
    add_shares(parties, "inverse_w_times_u", "dummy_sharing_of_one", "inverse_w_times_u_plus_one")
    # (w^(-1)*[u] + 1) * 2^(-1)
    inverse_two = modular_multiplicative_inverse(2, p)
    for party in parties:
        party.set_shares("dummy_sharing_of_inverse_two", inverse_two)
    multiply_shares(parties, "inverse_w_times_u_plus_one", "dummy_sharing_of_inverse_two", "temporary_random_bit")
    reset_parties(parties)
    # save the random bit
    for party in parties:
        party.set_random_number_bit_share_to_temporary_random_bit_share(bit_index)


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
    print("# po xorze gornego rzedu")
    for party in parties:
        party.print_test_z_tables()
    # Romb the bits
    for i in range(l - 1, -1, -1):
        # set x,y,X,Y
        print("# dane do romba")
        for party in parties:
            party.prepare_for_next_romb(i)
            party.print_test_2()
        ### x AND y
        multiply_shares(parties, "x", "y", "z")
        reset_parties(parties)
        print("# x AND y")
        for party in parties:
            party.print_test_1()
        ### X XOR Y
        xor_shares(parties, "X", "Y", "Z")
        reset_parties(parties)
        ### x AND (X XOR Y)
        multiply_shares(parties, "x", "Z", "Z")
        reset_parties(parties)
        ### x AND (X XOR Y) XOR X
        xor_shares(parties, "Z", "X", "Z")
        print("# x AND (X XOR Y) XOR X")
        for party in parties:
            party.print_test_1()
        reset_parties(parties)
    # calculate result
    # [res] = a_l XOR [r_l] XOR [Z]
    for party in parties:
        party.prepare_shares_for_res_xors(l, l)
    # a_l XOR [r_l] -> przypisz do [res]
    xor_shares(parties, "a_l", "r_l", "res")
    reset_parties(parties)
    # [res] XOR [Z] -> przypisz do [res]
    xor_shares(parties, "res", "Z", "res")
    print("a_l XOR r_l XOR Z")
    for party in parties:
        party.print_test_3()
    reset_parties(parties)
    # shares of the result are now stored in share named "res"


def main():
    # liczba bitów s,d = 3
    s = 4
    d = 6
    # liczba bitów d,s <= l
    l = 3
    # liczba pierwsza liczba bitów p = ?, p > a
    p = 61
    # liczba bitów p = l + k
    k = 1
    # liczba serwerow
    n = 5
    # serwery odzyskujace sekret
    t = 2

    # Create parties
    parties: list[Party] = []
    for i in range(n):
        party = Party(t, n, i + 1, p)
        parties.append(party)

    # Set the parties for each party
    for party in parties:
        party.set_parties(parties)

    # Calulate A for each party
    for party in parties:
        party.calculate_A()

    # r dlugosci l+k+1 bitow
    for i in range(l + k + 1):
        share_random_bit(parties, p, n, t, i)

    # test reconstruction of random bit shares
    for bit_index in range(l + k + 1):
        indeksy = [_ for _ in range(n)]
        wybrane_indeksy = []
        for _ in range(t):
            pom = random.choice(indeksy)
            wybrane_indeksy.append(pom)
            indeksy.remove(pom)
        selected_shares = [parties[i].get_random_number_bit_share(bit_index) for i in wybrane_indeksy]
        # print(selected_shares)
        coefficients = computate_coefficients(selected_shares, p)
        opened = reconstruct_secret(selected_shares, coefficients, p)
        print("opened bit = ", opened)


if __name__ == "__main__":
    main()
