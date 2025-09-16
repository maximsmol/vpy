from argparse import ArgumentParser
from codecs import escape_decode
from io import StringIO
from tokenize import generate_tokens

from vpy.lex import Lexer
from vpy.parse import Parser


def main() -> None:
    argp = ArgumentParser()
    _ = argp.add_argument("source")

    subp = argp.add_subparsers(dest="command", required=True)

    lexp = subp.add_parser("lex")
    _ = lexp.add_argument("--reference", action="store_true")

    parsep = subp.add_parser("parse")

    args = argp.parse_args()

    assert isinstance(args.source, str)
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
            p = Parser(lex=Lexer(data=source))
            print(p.parse())


if __name__ == "__main__":
    main()
