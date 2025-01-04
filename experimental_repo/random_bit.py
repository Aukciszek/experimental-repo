from random import randint

def break_into_shares(number, n, p):
    """
    Breaks a number into `n` random shares such that the sum of the shares
    modulo `p` equals the original number.

    Args:
        number (int): The number to be shared.
        n (int): The number of shares to create.
        p (int): A prime number for the field (modulo).

    Returns:
        list: A list of `n` shares.
    """
    # Generate n-1 random shares in the range [0, p-1]
    shares = [randint(0, p-1) for _ in range(n - 1)]
    
    # Compute the final share to ensure the sum of shares equals the original number (mod p)
    final_share = (number - sum(shares)) % p
    shares.append(final_share)
    print(f"Shares: {shares}")

    return shares

def binary_exponentiation(b, k, n):
    a = 1
    while k:
        if k & 1:
            a = (a * b) % n
        b = (b * b) % n
        k >>= 1
    return a

def inverse_modulo(b: int, n: int) -> int:
    """
    Returns the modular multiplicative inverse of b modulo n using the extended Euclidean algorithm.
    :param b: The number to find the inverse of.
    :param n: The modulus.
    :return: The modular multiplicative inverse of b modulo n.
    """
    A = n
    B = b
    U = 0
    V = 1
    while B != 0:
        q = A // B
        A, B = B, A - q * B
        U, V = V, U - q * V
    if U < 0:
        return U + n
    return U

def modular_sqrt(a, p) -> int:
    """
    Compute the square root of `a` modulo `p` using modular arithmetic.
    Works for prime `p` and assumes `a` is a quadratic residue modulo `p`.

    Args:
        a (int): The number whose square root is to be computed.
        p (int): The prime modulus.

    Returns:
        int: A square root of `a` modulo `p`, or None if no solution exists.
    """
    # Check if a solution exists using the Legendre symbol
    if pow(a, (p - 1) // 2, p) != 1:
        return None  # `a` is not a quadratic residue modulo `p`

    # Special case: p ≡ 3 (mod 4)
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)

    # General case: Tonelli-Shanks algorithm
    # Find `q` and `s` such that p-1 = q * 2^s with q odd
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    # Find a non-quadratic residue `z`
    z = 2
    while pow(z, (p - 1) // 2, p) == 1:
        z += 1

    # Initialize variables
    m = s
    c = pow(z, q, p)
    t = pow(a, q, p)
    r = pow(a, (q + 1) // 2, p)

    # Iterate to find the square root
    while t != 1:
        # Find the smallest i (0 < i < m) such that t^(2^i) ≡ 1 (mod p)
        t2i = t
        i = 0
        for i in range(1, m):
            t2i = pow(t2i, 2, p)
            if t2i == 1:
                break

        # Update variables
        b = pow(c, 2 ** (m - i - 1), p)
        r = (r * b) % p
        t = (t * b * b) % p
        c = (b * b) % p
        m = i

    return r

def get_shares(numbers, n, p):
    return [break_into_shares(x, n, p) for x in numbers]

def RandomBit(secrets, p):
    while True:
        U = get_shares(secrets, len(secrets), p)
        distributed_u = [[U[j][i] for j in range(len(U))] for i in range(len(U[0]))]
        print(f"Distributed U: {distributed_u}")

        V = []
        for s in distributed_u:
            secret_u = sum(s) % p
            secret_v = binary_exponentiation(secret_u, 2, p)
            V.append(secret_v)
        
        print(f"V: {V}")

        if sum(V) % p == 0:
            continue

        w = modular_sqrt(sum(V), p)
        if w is None:  # If no modular square root exists, retry
            print("No modular square root found; retrying...")
            continue

        one_bit_shares = [((inverse_modulo(w, p) * sum(x))) % p for x in distributed_u]
        print(f'One bit shares: {one_bit_shares}')

        one_bit = ((sum(one_bit_shares))+1)*(1/2)%p

        print(f'Random bit: {one_bit}')

        return one_bit

p = 23  # Example prime

for i in range(10):
    RandomBit([2225+i, 22246+i], p)