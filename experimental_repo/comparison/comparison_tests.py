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
    # print(f"shares_for_clients {shares_for_clients}")

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

    return parties, bits_of_r


def compare(n, t, d, s, p, k, l, r=None):
    #
    parties, bits_of_r = setup_parties(n, t, d, s, p, k, l, r)

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
    # print("opened_a = ", opened_a)
    a_bin = binary(opened_a)
    while len(a_bin) < l + k:
        a_bin.append(0)
    comparison_a_bits = a_bin
    print(f"a {opened_a} \t{comparison_a_bits}")
    print(f"r {r} \t{bits_of_r}")

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

    return result


# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


def compare_with_less_prints(n, t, d, s, p, k, l, r=None):
    block_print()
    result = compare(n, t, d, s, p, k, l, r)
    enable_print()
    # print(f"n: {n} | t: {t} | p: {p} | k: {k} | l: {l} | l+k+1: {l + k + 1}")
    # print(f"{d} >= {s} ? | expected = {expected} | wyn = {result}")
    # assert expected == result
    return result


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


def main3():
    l = 2
    k = 1
    p = 5
    licznik_zlych = 0
    for random_number in range(0, 2 ** l + k):
        for d in range(2 ** l):
            for s in range(2 ** l):
                expected_result = (d >= s)
                result = compare(n=3, t=1, d=d, s=s, p=p, k=k, l=l, r=random_number)
                if (expected_result != result % p):
                    licznik_zlych += 1
                    print("zle")
                print("\n", "-" * 100, "\n")
    print("liczniek_zlych", licznik_zlych)


def petla(n, t, p, k, l):
    print(f"n {n} t {t} l {l} k {k} p {p}")
    licznik_zlych = 0
    for random_number in range(0, 2 ** l + k):
        for d in range(2 ** l):
            for s in range(2 ** l):
                expected_result = (d >= s)
                result = compare_with_less_prints(n=n, t=t, d=d, s=s, p=p, k=k, l=l, r=random_number)
                if (expected_result != result % p):
                    licznik_zlych += 1
                    # print("zle")
                # print("\n", "-" * 100, "\n")
    print("licznik_zlych", licznik_zlych, "/", (2 ** l + k) * (2 ** l) * (2 ** l))


def main4():
    petla(n=3, t=1, p=5, k=1, l=2)
    petla(n=3, t=1, p=7, k=1, l=2)
    petla(n=3, t=1, p=13, k=1, l=3)
    petla(n=5, t=1, p=7, k=1, l=2)
    petla(n=5, t=2, p=7, k=1, l=2)
    petla(n=5, t=2, p=13, k=1, l=3)


if __name__ == "__main__":
    # main()
    # main2()
    # main3()
    main4()
