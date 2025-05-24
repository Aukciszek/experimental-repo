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
