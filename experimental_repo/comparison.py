import os
import random

binary_internal = lambda n: n > 0 and [n & 1] + binary_internal(n >> 1) or []

def binary(n):
    if n==0:
        return [0]
    else:
        return binary_internal(n)

def f(x, coefficients, p, t):
    return sum([coefficients[i] * x**i for i in range(t)]) % p

def Shamir(t, n, k0):
    p = 23

    coefficients = [random.randint(0, p - 1) for _ in range(t)]
    coefficients[0] = k0

    if coefficients[-1] == 0:
        coefficients[-1] = random.randint(1, p - 1)

    shares = []

    for i in range(1, n + 1):
        shares.append((i, f(i, coefficients, p, t)))

    return shares, p

def new_and(a,b):
    return a & b

def romb(x, X, y, Y):
    return (x & y, x & (X ^ Y) ^ X)


def main():
    s = 3
    d = 9

    k = 1
    l = 3

    r = []
    for i in range(l + k + 2):
        r.append(int.from_bytes(os.urandom(1)) % 2)

    print("r: ", r)

    r_prim = sum([r[i] * pow(2, i) for i in range(l + k + 2)])

    print("r_prim: ", r_prim)

    a = pow(2, l + k + 1) - r_prim + pow(2, l) + d - s
    if a < 0:
        print("a: ", a)
        print("a is negative")
        return

    print("a: ", a)

    a_bin = binary(a)
    r_prim_bin = binary(r_prim)

    while len(a_bin) < l + k + 2:
        a_bin.append(0)

    while len(r_prim_bin) < l + k + 2:
        r_prim_bin.append(0)

    print("a in binary: ", a_bin)
    print("r_prim in binary: ", r_prim_bin)

    z = []

    for i in range(l):
        z.append((a_bin[i] ^ r_prim_bin[i], a_bin[i]))

    z = list(reversed(z))
    z.append((0, 0))

    print("z: ", z)

    while len(z) > 1:
        z[0] = romb(z[0][0], z[0][1], z[1][0], z[1][1])
        z.pop(1)

    print("z: ", z)

    res = a_bin[l] ^ r_prim_bin[l] ^ z[0][1]

    print("res: ", res)

    assert res == 1


if __name__ == "__main__":
    main()
