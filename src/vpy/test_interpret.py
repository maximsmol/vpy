from vpy.interpret import Interpreter

from .lex import Lexer
from .parse import Parser


def main() -> None:
    src = "1 + 2 + 3"

    l = Lexer(data=src)
    p = Parser(lex=l)
    ast = p.parse_expr()
    print(src)

    print()
    print("Evaluated:")
    i = Interpreter()

    ours = i.eval(ast)
    reference = eval(src, globals={}, locals={})  # noqa: S307

    print(ours)
    print("Reference:")
    print(reference)

    if ours != reference:
        print("!!! Mismatch")
    else:
        print("Matches")

    cur = p.tok()
    if cur.type != "endmarker":
        print()
        print("!!! Leftover tokens:")
        while cur.type != "endmarker":
            print(cur)
            cur = p.tok()


if __name__ == "__main__":
    main()
