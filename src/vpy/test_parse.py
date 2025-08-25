import ast
import difflib
import tokenize
from pathlib import Path

from .lex import Lexer
from .parse import Parser


def process(src: str) -> bool:
    res = True

    l = Lexer(data=src)
    p = Parser(lex=l)
    ast_ours = p.parse()

    # todo(maximsmol): preserve missing newlines at end of input
    if src[-1] not in "\n\r":
        src += "\n"

    unparse = ast_ours.unparse()
    if unparse != src:
        print("!!! Unparse not equal")
        diffs = list(
            difflib.unified_diff(src.split("\n"), unparse.split("\n"), lineterm="")
        )

        res = False
        # todo(maximsmol): colorize this
        for l in diffs:
            print(l)

    # print(ast_ours)
    # print(ast.dump(ast_ours.to_ast(), indent=2))
    ast_ours_lines = ast.dump(ast_ours.to_ast(), indent=2).split("\n")

    cur = p.tok()
    if cur.type != "endmarker":
        # print()
        print("!!! Leftover tokens:")
        res = False
        while cur.type != "endmarker":
            print(cur)
            cur = p.tok()

    ast_reference = ast.parse(src)
    # print(ast.dump(ast_reference, indent=2))
    ast_reference_lines = ast.dump(ast_reference, indent=2).split("\n")

    diffs = list(difflib.unified_diff(ast_ours_lines, ast_reference_lines, lineterm=""))
    if len(diffs) != 0:
        res = False
        # todo(maximsmol): colorize this
        for l in diffs:
            print(l)

    return res


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
