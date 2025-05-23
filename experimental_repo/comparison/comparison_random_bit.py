from comparison_tests import *
from experimental_repo.comparison.random_bit import share_random_bit


def setup_parties(n, t, d, s, p, k, l):
    """
    Args:
        n: liczba serwerow
        t: liczba serwerow odzyskujacych sekret
        d: pierwsza liczba do porownania
        s: druga liczba do porownania
        p: liczba pierwsza (liczba bitów p = l + k)
        k: parametr dodatkowy, liczba bitów p = l + k
        l: liczba bitów s,d <= l
    """

    shares_d = Shamir(t, n, d, p)
    print("d:", d, f"| shares_d: {shares_d}")
    shares_s = Shamir(t, n, s, p)
    print("s:", s, "| shares_s:", shares_s)

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

    # r dlugosci l+k+1 bitow
    # od najnizszej potegi
    for i in range(l + k + 1):
        share_random_bit(parties, p, n, t, i)

    for party in parties:
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

    return parties


def compare(n, t, d, s, p, k, l):
    parties = setup_parties(n, t, d, s, p, k, l)

    # Calculate comparison a
    # [a] = 2^(l+k+1) - [r] + 2^l + [d] - [s]
    # first share   --> d_share
    # second share  --> s_share
    for party in parties:
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

    # Compare s,d and save the resulting shares in share named "res"
    # d >= s  -->  res=1
    # d < s   -->  res=0
    comparison(parties, opened_a, l, k)

    # get comparison result
    result = reconstruct_from_shares_by_share_name(n, t, p, parties, "res")
    print("wynik porównania:", result)

    return result


if __name__ == "__main__":
    compare(n=5, t=2, d=5, s=3, p=61, k=1, l=3)
