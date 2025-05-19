import os
import sys

from comparison import *


def setup_parties(n, t, d, s, p, k, l, r=None):
    """
    Args:
        n: liczba serwerow
        t: liczba serwerow odzyskujacych sekret
        d: pierwsza liczba do porownania
        s: druga liczba do porownania
        p: liczba pierwsza (liczba bitów p = l + k)
        k: parametr dodatkowy, liczba bitów p = l + k
        l: liczba bitów s,d <= l
        r: wmieszywana liczba, domyślnie losowana
    """

    shares_d = Shamir(t, n, d, p)
    print("d:", d, f"| shares_d: {shares_d}")
    shares_s = Shamir(t, n, s, p)
    print("s:", s, "| shares_s:", shares_s)

    # r dlugosci l+k+2 bitow
    # od najnizszej potegi
    if r is None:
        bits_of_r = [random.randint(0, 1) for _ in range(l + k + 2)]
    else:
        bits_of_r = binary(r)
        while len(bits_of_r) < l + k + 2:
            bits_of_r.append(0)
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

    return parties


def compare(n, t, d, s, p, k, l, r=None):
    #
    parties = setup_parties(n, t, d, s, p, k, l, r)
    real_a = (
            pow(2, l + k + 1)
            - r
            + pow(2, l)
            + d
            - s
    )

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

    print(a_comparison_share)
    selected_shares = [a_comparison_share[random.randint(1, n) - 1]]
    print(selected_shares)
    print(selected_shares[0][1] % p)
    coefficients = computate_coefficients(selected_shares, p)
    opened_a = reconstruct_secret(selected_shares, coefficients, p)
    print("opened_a = ", opened_a)

    # Compare s,d and save the resulting shares in share named "res"
    # d >= s  -->  res=1
    # d < s   -->  res=0
    comparison(parties, opened_a, l, k)

    # Reconstruct res
    selected_shares = [(i, parties[i].get_share_by_name("res")) for i in range(n)]
    coefficients = computate_coefficients(selected_shares, p)
    result = reconstruct_secret(selected_shares, coefficients, p)

    print("wynik porównania:", result)

    return result, real_a, opened_a


# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


def compare_with_less_prints(n, t, d, s, p, k, l, r=None):
    block_print()
    expected = 1 if d >= s else 0
    result, ra, oa = compare(n, t, d, s, p, k, l, r)
    enable_print()
    # print(f"n: {n} | t: {t} | p: {p} | k: {k} | l: {l} | l+k+1: {l + k + 1}")
    # print(f"{d} >= {s} ? | expected = {expected} | wyn = {result}")
    # assert expected == result
    return expected, result


def main():
    p = 61
    counter = 0
    for i in range(0, 2 ** 6):
        e1, r1 = compare_with_less_prints(n=3, t=1, d=5, s=6, p=61, k=1, l=3, r=i)
        e2, r2 = compare_with_less_prints(n=3, t=1, d=6, s=5, p=61, k=1, l=3, r=i)
        if e1 % p != r1 % p or e2 % p != r2 % p:
            # print(i,binary(i))
            counter += 1
    print(counter, 2 ** 6)

    counter = 0
    for i in range(0, 2 ** 6):
        e1, r1 = compare_with_less_prints(n=3, t=1, d=0, s=7, p=61, k=1, l=3, r=i)
        e2, r2 = compare_with_less_prints(n=3, t=1, d=7, s=0, p=61, k=1, l=3, r=i)
        if e1 % p != r1 % p or e2 % p != r2 % p:
            # print(i, binary(i))
            counter += 1
    print(counter, 2 ** 6)


def main2():
    e1, r1 = compare_with_less_prints(n=5, t=2, d=5, s=6, p=61, k=1, l=3, r=50)
    print(e1 % 61, r1 % 61)
    e1, r1 = compare_with_less_prints(n=5, t=2, d=6, s=5, p=61, k=1, l=3, r=50)
    print(e1 % 61, r1 % 61)


if __name__ == "__main__":
    main()
    main2()
