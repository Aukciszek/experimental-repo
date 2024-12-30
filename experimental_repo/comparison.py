import os

binary = lambda n: n > 0 and [n & 1] + binary(n >> 1) or []


def romb(x, X, y, Y):
    return (x & y, x & (X ^ Y) ^ X)


for _ in range(10000000):
    s = 3
    d = 9

    k = 1
    l = 3

    r = []
    for i in range(l + k + 2):
        r.append(int.from_bytes(os.urandom(1)) % 2)

    # print("r: ", r)

    r_prim = sum([r[i] * pow(2, i) for i in range(l + k + 2)])

    # print("r_prim: ", r_prim)

    a = pow(2, l + k + 1) - r_prim + pow(2, l) + d - s
    if a < 0:
        continue

    # print("a: ", a)

    a = abs(a)

    # print("a_abs: ", a)

    a_bin = binary(a)
    r_prim_bin = binary(r_prim)

    while len(a_bin) < l + k + 2:
        a_bin.append(0)

    while len(r_prim_bin) < l + k + 2:
        r_prim_bin.append(0)

    # print("a in binary: ", a_bin)
    # print("r_prim in binary: ", r_prim_bin)

    z = []

    for i in range(l):
        z.append((a_bin[i] ^ r_prim_bin[i], a_bin[i]))

    z = list(reversed(z))
    z.append((0, 0))

    # print("z: ", z)

    while len(z) > 1:
        z[0] = romb(z[0][0], z[0][1], z[1][0], z[1][1])
        z.pop(1)

    # print("z: ", z)

    res = a_bin[l] ^ r_prim_bin[l] ^ z[0][1]

    # print("res: ", res)

    assert res == 1
