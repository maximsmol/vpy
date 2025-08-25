import difflib
import tokenize
from pathlib import Path
from tokenize import TokenInfo, generate_tokens
from traceback import print_exc

from .lex import Lexer, Token


def process(src: str) -> bool:
    toks_ours: list[Token] = []
    l = Lexer(data=src)
    while True:
        try:
            tok = l.next()
        except RuntimeError:
            print_exc()
            break

        toks_ours.append(tok)
        if tok.type == "endmarker":
            break

    toks_ours_lines = [str(x.token_info()) for x in toks_ours if x.type != "whitespace"]

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

    diffs = list(
        difflib.unified_diff(toks_ours_lines, toks_reference_lines, lineterm="")
    )
    if len(diffs) == 0:
        return True

    # todo(maximsmol): colorize this
    for l in diffs:
        print(l)

    return False


def main() -> None:
    assert process("1 + 2")
    assert process("1 + 2 + 3")

    root_p = Path(tokenize.__file__).parent

    for f in [Path(tokenize.__file__)]:
        print(f">>> {f.relative_to(root_p)}:")
        ok = process(f.read_text())
        if not ok:
            break


if __name__ == "__main__":
    main()
