import os
import sys

from comparison import *


def reconstruct_from_shares_by_share_name(n, t, p, parties, share_name: str):
    indeksy = [_ for _ in range(n)]
    wybrane_indeksy = []
    for _ in range(t):
        w = random.choice(indeksy)
        wybrane_indeksy.append(w)
        indeksy.remove(w)
    selected_shares = [(i + 1, parties[i].get_share_by_name(share_name)) for i in wybrane_indeksy]
    coefficients = computate_coefficients(selected_shares, p)
    opened = reconstruct_secret(selected_shares, coefficients, p)
    print(f"opened {share_name} = {opened}")

    return opened


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

    # r dlugosci l+k+1 bitow
    # od najnizszej potegi
    if r is None:
        bits_of_r = [random.randint(0, 1) for _ in range(l + k + 1)]
    else:
        bits_of_r = binary(r)
        while len(bits_of_r) < l + k + 1:
            bits_of_r.append(0)
    oryginalne_r = sum([bits_of_r[i] * pow(2, i) for i in range(l + k + 1)])
    print(f"r: {oryginalne_r} | bits of r: {bits_of_r}")

    shares_of_bits_of_r = []
    for i in range(l + k + 1):
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

    # # test reconstruction of compared numbers
    # reconstruct_from_shares_by_share_name(n, t, p, parties, "s_share")

    # # test reconstruction of random bit shares
    # for i in range(len(shares_of_bits_of_r)):
    #     bit = shares_of_bits_of_r[i]
    #     selected_shares = [parties[0].get_random_number_bit_share(i), parties[3].get_random_number_bit_share(i)]
    #     print(selected_shares)
    #     coefficients = computate_coefficients(selected_shares, p)
    #     opened = reconstruct_secret(selected_shares, coefficients, p)
    #     print(bit, "opened = ", opened)

    return parties, bits_of_r


def compare(n, t, d, s, p, k, l, r=None):
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
    # reconstruct a
    coefficients = computate_coefficients(selected_shares, p)
    opened_a = reconstruct_secret(selected_shares, coefficients, p)
    # get bits of a
    a_bin = binary(opened_a)
    while len(a_bin) < l + k + 1:
        a_bin.append(0)
    comparison_a_bits = a_bin
    print(f"a {opened_a} \t{comparison_a_bits}")
    print(f"r {r} \t{bits_of_r}")

    # Compare s,d and save the resulting shares in share named "res"
    # d >= s  -->  res=1
    # d < s   -->  res=0
    comparison(parties, opened_a, l, k)

    # get comparison result
    result = reconstruct_from_shares_by_share_name(n, t, p, parties, "res")
    print("wynik porównania:", result)

    return result


def block_print():
    sys.stdout = open(os.devnull, 'w')


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


def main2():
    result = compare(n=5, t=2, d=5, s=6, p=61, k=1, l=3, r=30)
    print(f"expected = {int(5 >= 6)}, result = {result}")


def petla(n, t, p, k, l):
    print(f"n {n} t {t} l {l} k {k} p {p}")
    # 2^(l+k+1) - r + 2^l + d - s
    duza_potega = 2 ** (l + k + 1)
    # jesli random number jest wiekszy niz duza potega, a moze byc ujemne
    liczba_bitow_random_number = l + k + 1
    zakres_random_number = (0, 2 ** liczba_bitow_random_number - 1)
    mala_potega = 2 ** l
    zakres_porownywanych_liczb = (0, mala_potega - 1)
    # a (i niezaciemnione a?) musi byc mniejsze niz p inaczej nie wychodzi
    zakres_niezaciemnionego_a = (
        (
                duza_potega
                + mala_potega
                + zakres_porownywanych_liczb[0]
                - zakres_porownywanych_liczb[1]
        ),
        (
                duza_potega
                + mala_potega
                + zakres_porownywanych_liczb[1]
                - zakres_porownywanych_liczb[0]
        )
    )
    zakres_a = (
        zakres_niezaciemnionego_a[0] - zakres_random_number[1],
        zakres_niezaciemnionego_a[1] - zakres_random_number[0])
    print(
        f"zakres niezaciemnionego a {zakres_niezaciemnionego_a}, zakres random number {zakres_random_number}, zakres a {zakres_a}")
    if p <= zakres_a[1]:
        print(f"uwaga, liczba pierwsza moze byc mniejsza niz a: {p} <= {zakres_a[1]}")

    licznik = 0
    licznik_zlych = 0
    for random_number in range(zakres_random_number[0], zakres_random_number[1] + 1):
        for d in range(zakres_porownywanych_liczb[0], zakres_porownywanych_liczb[1] + 1):
            for s in range(zakres_porownywanych_liczb[0], zakres_porownywanych_liczb[1] + 1):
                expected_result = (d >= s)
                result = compare_with_less_prints(n=n, t=t, d=d, s=s, p=p, k=k, l=l, r=random_number)
                if (expected_result != result % p):
                    licznik_zlych += 1
                    # print("zle")
                licznik += 1
                # print("\n", "-" * 100, "\n")
    print("licznik_zlych", licznik_zlych, "/", licznik)


def main4():
    # petla(n=3, t=1, p=29, k=1, l=3)

    petla(n=5, t=2, p=43, k=1, l=3)
    petla(n=5, t=2, p=47, k=1, l=3)
    petla(n=5, t=2, p=53, k=1, l=3)
    petla(n=5, t=2, p=61, k=1, l=3)


if __name__ == "__main__":
    # main()
    main4()
    print("\n", "-" * 200, "\n")
    main2()
