import os
import math

binary_internal = lambda n: n > 0 and [n & 1] + binary_internal(n >> 1) or []

def binary(n):
    wyn=None
    if n==0:
        wyn=[0]
    else:
        wyn=binary_internal(n)
    return wyn 

def binary_ZU2(x):
    wyn=None
    if x>=0:
        wyn=binary(x)+[0]
    elif x<0:
        dl=math.ceil((math.log2(abs(x))))+1
        pot = pow(2,dl)
        wyn=[int(_) for _ in bin(pot+x)[2:].zfill(dl-1)]
        wyn.reverse()
        if len(wyn)!=dl or x==-1:
            wyn+=[1]
    return wyn

def binary_ZU2_fixed_length(x,dl):
    wyn=None
    if x>=0:
        wyn=binary(x)+[0]
        while len(wyn)<dl:
            wyn.append(0)
    elif x<0:
        #dl=math.ceil((math.log2(abs(x))))+1
        pot = pow(2,dl)
        wyn=[int(_) for _ in bin(pot+x)[2:].zfill(dl-1)]
        wyn.reverse()
        if len(wyn)!=dl or x==-1:
            wyn+=[1]
    return wyn

# print("binary")
# print(0,binary(0))
# print(1,binary(1))
# print(2,binary(2))

# print("binary ZU2")
# for i in range(-9,10):
#     print(i,binary_ZU2(i))

def romb(x, X, y, Y):
    return (x & y, x & (X ^ Y) ^ X)

def main():
    
    for _ in range(100):
        r = []
        for i in range(40):
            r.append(int.from_bytes(os.urandom(1)) % 2)
        r_prim = sum([r[i] * pow(2, i) for i in range(40)]) - r[i] * pow(2, 40)
        assert r==binary_ZU2_fixed_length(r_prim,40)

    for _ in range(1000):
        s = 3
        d = 9

        k = 1
        l = 3

        r = []
        for i in range(l + k + 2):
            r.append(int.from_bytes(os.urandom(1)) % 2)

        #print("r: ", r)

        r_prim = sum([r[i] * pow(2, i) for i in range(l + k + 1)]) - r[i] * pow(2, l + k + 1)

        #print("r_prim: ", r_prim)

        a = pow(2, l + k + 1) - r_prim + pow(2, l) + d - s
        if a < 0:
            print("a: ", a)
            print("a is negative")
            return

        #print("a: ", a)

        a_bin = binary(a)
        r_prim_bin = binary_ZU2_fixed_length(r_prim,l + k + 2)

        while len(a_bin) < l + k + 2:
            a_bin.append(0)

        while len(r_prim_bin) < l + k + 2:
            print("XD")
            r_prim_bin.append(0)

        #print("a in binary: ", a_bin)
        #print("r_prim in binary: ", r_prim_bin)

        z = []

        for i in range(l):
            z.append((a_bin[i] ^ r_prim_bin[i], a_bin[i]))

        z = list(reversed(z))
        z.append((0, 0))

        #print("z: ", z)

        while len(z) > 1:
            z[0] = romb(z[0][0], z[0][1], z[1][0], z[1][1])
            z.pop(1)

        #print("z: ", z)

        res = a_bin[l] ^ r_prim_bin[l] ^ z[0][1]

        #print("res: ", res)

        assert res==1
    


if __name__ == "__main__":
    main()
