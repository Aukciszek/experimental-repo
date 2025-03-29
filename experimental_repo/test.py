def multiply_bit_shares_by_powers_of_2(shares):
    multiplied_shares = []
    for i in range(len(shares)):
        multiplied_shares.append(2**i * shares[i][1])
    return multiplied_shares

def add_multiplied_shares(multiplied_shares):
    share_r = multiplied_shares[0]
    for i in range(1, len(multiplied_shares)):
        share_r += multiplied_shares[i]
    return share_r

tab = [0,1,0,1,1,1]
tab2 = [(9,i) for i in tab]

pom = multiply_bit_shares_by_powers_of_2(tab2)
pom = add_multiplied_shares(pom)

print(pom)