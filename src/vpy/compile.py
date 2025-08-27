from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .parse import (
    AExpr,
    AndExpr,
    AstLiteral,
    Atom,
    ExpressionStmt,
    FileInput,
    MExpr,
    Node,
    OrExpr,
    Power,
    Primary,
    ShiftExpr,
    SimpleStmt,
    StarredExpression,
    Statement,
    StmtList,
    UExpr,
    XorExpr,
)


@dataclass
class Compiler:
    var_idx: int = 1
    indent_level: int = 0
    lines: list[str] = field(default_factory=list)

    @contextmanager
    def indent(self) -> Generator[None]:
        try:
            self.indent_level += 1
            yield
        finally:
            self.indent_level -= 1

    def next_var(self) -> int:
        res = self.var_idx
        self.var_idx += 1
        return res

    def emit(self, x: str) -> None:
        self.lines.append("  " * self.indent_level + x)

    def compile_expr(self, x: Node) -> None:
        if isinstance(x, StarredExpression):
            return self.compile_expr(x.x)

        if isinstance(x, OrExpr):
            return self.compile_expr(x.rhs)

        if isinstance(x, XorExpr):
            return self.compile_expr(x.rhs)

        if isinstance(x, AndExpr):
            return self.compile_expr(x.rhs)

        if isinstance(x, ShiftExpr):
            return self.compile_expr(x.rhs)

        if isinstance(x, AExpr):
            if x.lhs is None:
                return self.compile_expr(x.rhs)

            assert x.op.text == "+"

            lhs = self.compile_expr(x.lhs)
            rhs = self.compile_expr(x.rhs)

            res = self.next_var()
            self.emit(f"%{res} = add i64 %{lhs}, %{rhs}")

            return res

        if isinstance(x, MExpr):
            return self.compile_expr(x.rhs)

        if isinstance(x, UExpr):
            return self.compile_expr(x.x)

        if isinstance(x, Power):
            return self.compile_expr(x.x)

        if isinstance(x, Primary):
            return self.compile_expr(x.x)

        if isinstance(x, Atom):
            return self.compile_expr(x.x)

        if isinstance(x, AstLiteral):
            assert x.x.type == "decinteger"

            res = self.next_var()
            self.emit(f"%{res} = bitcast i64 {x.x.text} to i64")
            return res

        raise NotImplementedError(f"unknown node: {x.type}")

    def compile(self, x: Node) -> None:
        if isinstance(x, FileInput):
            self.emit("define i64 @test() {")
            with self.indent():
                for s in x.xs:
                    self.compile(s)
            self.emit("}")
            return

        if isinstance(x, Statement):
            self.compile(x.x)
            return

        if isinstance(x, StmtList):
            for s in x.xs:
                self.compile(s)
            return

        if isinstance(x, SimpleStmt):
            self.compile(x.x)
            return

        if isinstance(x, ExpressionStmt):
            res = self.compile_expr(x.x)
            self.emit("")
            self.emit(f"ret i64 %{res}")
            return

        raise NotImplementedError(f"unknown node: {x.type}")
