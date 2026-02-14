from argparse import ArgumentParser
import ast
from codecs import escape_decode
from io import StringIO
from pathlib import Path
from tokenize import generate_tokens

from vpy.lex import Lexer
from vpy.parse import Parser


def main() -> None:
    argp = ArgumentParser()
    _ = argp.add_argument("source")
    _ = argp.add_argument("--file", action="store_true")

    subp = argp.add_subparsers(dest="command", required=True)

    lexp = subp.add_parser("lex")
    _ = lexp.add_argument("--reference", action="store_true")

    parsep = subp.add_parser("parse")
    _ = parsep.add_argument("--reference", action="store_true")

    args = argp.parse_args()

    assert isinstance(args.source, str)
    if args.file:
        source = Path(args.source).read_text(encoding="utf-8")
    else:
        source = escape_decode(args.source)[0].decode()

    match args.command:
        case "lex":
            if args.reference:
                io = StringIO(source)
                for x in generate_tokens(io.readline):
                    print(x)

                return

            l = Lexer(data=source)
            while True:
                x = l.next()
                print(x)

                if x.type == "endmarker":
                    break

        case "parse":
            if args.reference:
                print(ast.dump(ast.parse(source), indent=2))

                return

            p = Parser(lex=Lexer(data=source))
            print(p.parse())


if __name__ == "__main__":
    main()
