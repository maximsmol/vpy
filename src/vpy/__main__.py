from .lex import Lexer
from .parse import Parser


def main() -> None:
    src = "1 + 2"

    l = Lexer(data=src)
    p = Parser(lex=l)
    print(p.file_input())

    cur = p.tok()
    if cur.type == "endmarker":
        return

    print()
    print(">>> Leftover tokens:")
    while cur.type != "endmarker":
        print(cur)
        cur = p.tok()


if __name__ == "__main__":
    main()
