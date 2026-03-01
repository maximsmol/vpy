import ast
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from types import CodeType
from typing import Any, Literal, overload

from .interpret import Interpreter, Scope
from .lex import Lexer
from .parse import ExpressionStmt, FileInput, Parser, Statement
from .values import VpyValue


@dataclass(kw_only=True)
class RefEvalRes:
    value: object
    globals: dict[str, object]
    locals: dict[str, object]


def reference_eval(src: str) -> RefEvalRes:
    ast_ref = ast.parse(src)
    stmts = ast_ref.body

    globals_ref = {}
    locals_ref = {}

    def eval_(x: str | CodeType) -> Any:
        return eval(x, globals=globals_ref, locals=locals_ref)  # noqa: S307

    if isinstance(stmts[-1], ast.Expr):
        eval_(compile_(stmts[:-1]))
        value = eval_(compile_(stmts[-1].value, mode="eval"))
    else:
        eval_(compile_(src))
        value = None

    return RefEvalRes(value=value, globals=globals_ref, locals=locals_ref)


@overload
def compile_(
    x: str | list[ast.stmt] | ast.Module, *, mode: Literal["exec"] = "exec"
) -> CodeType: ...


@overload
def compile_(x: str | ast.expr, *, mode: Literal["eval"]) -> CodeType: ...


def compile_(
    x: str | list[ast.stmt] | ast.expr | ast.Module,
    *,
    mode: Literal["exec", "eval"] = "exec",
) -> CodeType | ast.AST:
    src = x
    if isinstance(src, ast.expr):
        src = ast.Expression(body=src)
    if isinstance(src, list):
        src = ast.Module(body=src)

    return compile(src, filename="<inline>", mode=mode)


def process(src: str) -> bool:
    res = True

    l = Lexer(data=src)
    p = Parser(lex=l)
    ast_ours = p.parse()

    cur = p.tok()
    if cur.type != "endmarker":
        print("!!! Leftover tokens:")
        while cur.type != "endmarker":
            print(cur)
            cur = p.tok()

        res = False

    global_i = Interpreter()
    i = Scope(interpreter=global_i)

    assert isinstance(ast_ours, FileInput)

    assert isinstance(ast_ours.xs[-1], Statement)
    if isinstance(ast_ours.xs[-1].x.xs[-1].x, ExpressionStmt):
        for stmt in ast_ours.xs[:-1]:
            i.exec(stmt)

        ours = i.eval(ast_ours.xs[-1].x.xs[-1].x.x)
    else:
        i.exec(ast_ours)
        ours = VpyValue.from_none()

    reference = reference_eval(src)

    ours_py = ours.interpreted_to_python(global_i)
    if type(ours_py) is not type(reference.value) or ours_py != reference.value:
        print("!!! Mismatch")
        print("Ours:")
        print(ours_py)
        print()
        print("Reference:")
        print(reference.value)
        print()
        res = False

    locals_ref_set = set(reference.locals.keys())
    locals_set = set(i.locals.keys())
    if locals_ref_set > locals_set:
        print("!!! Missing locals:")
        for x in locals_ref_set - locals_set:
            print(f"- {x}")
        print()
        res = False

    if locals_set > locals_ref_set:
        print("!!! Extraneous locals:")
        for x in locals_set - locals_ref_set:
            print(f"- {x}")
        print()
        res = False

    # todo(maximsmol): check globals
    for k, ref in reference.locals.items():
        if k not in i.locals:
            continue

        # todo(maximsmol): also check functions
        if type(ref) is type(main):
            continue

        ours = i.locals[k]
        ours_py = ours.interpreted_to_python(global_i)
        if type(ours_py) is not type(ref) or ours_py != ref:
            print("!!! Mismatch")
            print("Ours:")
            print(ours_py)
            print()
            print("Reference:")
            print(ref)
            print()
            res = False

    return res


def main() -> None:
    assert process("1 + 2 + 3")
    assert process("a = 123")
    assert process(
        dedent("""
            a = 123
            a + 1
        """)[1:]
    )
    assert process("10 % 6")
    assert process("1 == 1")
    assert process("2 * 5")
    assert process(
        dedent("""
            a = 1
            a += 2
        """)[1:]
    )
    assert process("True")
    assert process(
        dedent("""
            a = 10
            if False:
                a = 20
            a
        """)[1:]
    )
    assert process(
        dedent("""
            a = 2
            while a <= 10:
                a = a * a
            a
        """)[1:]
    )
    assert process(
        dedent("""
            def f(x: int) -> int:
                return x + 10

            f(10)
        """)[1:]
    )
    assert process(
        dedent("""
            def f(x: int) -> int | float:
                if x == 5:
                    return 123
                return 0.999

            f(10)
        """)[1:]
    )

    print("Smoketest OK")

    root_p = Path(__file__).parent.parent.parent / "tests/problems_99"

    for f in root_p.iterdir():
        print(f">>> {f.relative_to(root_p)}:")
        ok = process(f.read_text())
        if not ok:
            break
        print("  OK")


if __name__ == "__main__":
    main()
