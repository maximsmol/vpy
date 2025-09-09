import ast
from types import CodeType
from typing import Any, Literal, overload

from vpy.interpret import Interpreter

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
    src = "a = 123\na + 10"

    l = Lexer(data=src)
    p = Parser(lex=l)
    ast_ours = p.parse()
    print(src)

    print()
    print("Evaluated:")
    i = Interpreter()

    assert isinstance(ast_ours, FileInput)

    assert isinstance(ast_ours.xs[-1], Statement)
    if isinstance(ast_ours.xs[-1].x.xs[-1].x, ExpressionStmt):
        for stmt in ast_ours.xs[:-1]:
            i.exec(stmt)

        ours = i.eval(ast_ours.xs[-1].x.xs[-1].x.x)
    else:
        i.exec(ast_ours)
        ours = None

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
        eval_(src)
        reference = None

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

    print()

    print("Locals:")
    print(i.locals)
    print("Reference:")
    print(locals_ref)


if __name__ == "__main__":
    main()
