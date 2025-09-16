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
    smoke_test = True
    # smoke_test = False
    if smoke_test:
        assert process("1 + 2")
        assert process("1 + 2 + 3")
        assert process("a = 123")
        assert process("a = 123\na + 1")
        assert process("10 % 6")
        assert process("1 == 1")
        assert process("2 * 5")
        assert process("a = 1\na += 2")
        assert process("True")
        assert process("a = 10\nif False:\n    a = 20\na")

        print("Smoketest OK")

    root_p = Path(__file__).parent.parent.parent / "tests/problems_99"

    for f in root_p.iterdir():
        print(f">>> {f.relative_to(root_p)}:")
        ok = process(f.read_text())
        if not ok:
            break

    return

    root_p = Path(tokenize.__file__).parent

    for f in [Path(tokenize.__file__)]:
        print(f">>> {f.relative_to(root_p)}:")
        ok = process(f.read_text())
        if not ok:
            break


if __name__ == "__main__":
    main()
