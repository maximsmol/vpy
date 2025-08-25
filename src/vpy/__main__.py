from .lex import Lexer
from .parse import Parser


def main() -> None:
    src = "1 + 2 + 3"

    l = Lexer(data=src)
    p = Parser(lex=l)
    ast_ours = p.parse()
    print(ast_ours)

    print()
    print("Unparsed:")
    print(ast_ours.unparse(parens=True))

    cur = p.tok()
    if cur.type != "endmarker":
        print("!!! Leftover tokens:")
        while cur.type != "endmarker":
            print(cur)
            cur = p.tok()


if __name__ == "__main__":
    main()
