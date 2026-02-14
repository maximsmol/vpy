import ast
from pathlib import Path
from textwrap import dedent
from types import CodeType
from typing import Any, Literal, overload

from vpy.interpret import Interpreter, Scope, Value

from .lex import Lexer
from .parse import ExpressionStmt, FileInput, Parser, Statement


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


def main() -> None:
    # src = "1 + 2 + 3"
    # src = "a = 123"
    # src = dedent("""
    #     a = 123
    #     a + 1
    # """)[1:]
    # src = "10 % 6"
    # src = "1 == 1"
    # src = "2 * 5"
    # src = dedent("""
    #     a = 1
    #     a += 2
    # """)[1:]
    # src = "True"
    # src = dedent("""
    #     a = 10
    #     if False:
    #         a = 20
    #     a
    # """)[1:]
    # src = dedent("""
    #     a = 2
    #     while a <= 10:
    #         a = a * a
    #     a
    # """)[1:]
    # src = dedent("""
    #     def f(x: int) -> int:
    #         return x + 10

    #     f(10)
    # """)[1:]
    src = dedent("""
        def f(x: int) -> int | float:
            if x == 5:
                return 123
            return 0.999

        f(10)
    """)[1:]
    # src = Path("tests/problems_99/p2.01_is_prime.py").read_text(encoding="utf-8")
    # src = Path("tests/problems_99/p2.07_gcd.py").read_text(encoding="utf-8")

    l = Lexer(data=src)
    p = Parser(lex=l)
    ast_ours = p.parse()
    print(src)

    print()
    print("Evaluated:")
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
        ours = Value.from_none()

    ast_ref = ast.parse(src)
    stmts = ast_ref.body

    globals_ref = {}
    locals_ref = {}

    def eval_(x: str | CodeType) -> Any:
        return eval(x, globals=globals_ref, locals=locals_ref)  # noqa: S307

    if isinstance(stmts[-1], ast.Expr):
        eval_(compile_(stmts[:-1]))
        reference = eval_(compile_(stmts[-1].value, mode="eval"))
    else:
        eval_(compile_(src))
        reference = None

    print(ours)

    ours_py = ours.to_python(global_i)
    print(".to_python():")
    print(ours_py)

    print()
    print("Reference:")
    print(reference)

    if type(ours_py) is not type(reference) or ours_py != reference:
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

    print()

    print("Locals:")
    print(i.locals)
    print("Reference:")
    print(locals_ref)


if __name__ == "__main__":
    main()
