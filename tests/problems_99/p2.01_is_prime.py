def is_prime(x: int) -> bool:
    if x % 2 == 0:
        return False

    cur = 3
    while cur * cur < x:
        if x % cur == 0:
            return False
        cur += 2

    return True
