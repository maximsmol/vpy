import difflib
from tokenize import TokenInfo, generate_tokens

from .lex import Lexer, Token


def main() -> None:
    # todo(maximsmol): deal with encodings
    src = "1 + 2"

    toks_ours: list[Token] = []
    l = Lexer(data=src)
    while True:
        tok = l.next()
        toks_ours.append(tok)
        print(tok)
        if tok.type == "endmarker":
            break

    toks_ours_lines = [str(x.token_info()) for x in toks_ours if x.type != "whitespace"]

    print(">>>")

    # >>> Basic testing

    ran = False

    def readline() -> str:
        nonlocal ran

        if ran:
            return ""

        ran = True
        return src

    def token_info_str(x: TokenInfo) -> str:
        return str(
            TokenInfo(type=x.type, string=x.string, start=x.start, end=x.end, line="")
        )

    toks_reference = list(generate_tokens(readline))
    toks_reference_lines = [token_info_str(x) for x in toks_reference]

    for l in difflib.unified_diff(toks_ours_lines, toks_reference_lines):
        print(l)


if __name__ == "__main__":
    main()
