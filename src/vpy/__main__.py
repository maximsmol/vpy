import tokenize
from .lex import Lexer


def main():
    ran = False

    def readline() -> bytes:
        nonlocal ran

        if ran:
            return b""

        ran = True
        return b"1 + 2"

    for t in tokenize.tokenize(readline):
        print(t)

    return

    src = "1 + 2"

    l = Lexer(data=src)
    while True:
        print(l.next())


if __name__ == "__main__":
    main()
