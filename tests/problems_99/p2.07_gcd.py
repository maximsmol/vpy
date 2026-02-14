def gcd(a: int, b: int) -> int:
    while True:
        if a == 0:
            return b
        if b == 0:
            return a

        r = a % b
        a = b
        b = r


gcd(36, 63)
