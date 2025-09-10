import re
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from vpy.lex import Token

from .parse import (
    AExpr,
    AndExpr,
    AndTest,
    AssignmentStmt,
    AstLiteral,
    Atom,
    AugmentedAssignmentStmt,
    Comparison,
    ConditionalExpression,
    ExpressionList,
    ExpressionStmt,
    FileInput,
    MExpr,
    Node,
    NotTest,
    OrExpr,
    OrTest,
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

llvm_id_re = re.compile(r"^[-a-zA-Z$._][-a-zA-Z$._0-9]*$")


def llvmstr(x: str) -> str:
    x = x.replace('"', f"\\{ord('"'):x}")
    return f'"{x}"'


@dataclass
class Compiler:
    var_idx: int = 1
    indent_level: int = 0
    lines: list[str] = field(default_factory=list)

    locals: dict[str, str] = field(default_factory=dict)

    @contextmanager
    def indent(self) -> Generator[None]:
        try:
            self.indent_level += 1
            yield
        finally:
            self.indent_level -= 1

    def next_var(self, *, prefix: str = "") -> str:
        idx = self.var_idx
        self.var_idx += 1

        # note: we deliberately do not use unnamed variables (and use `.{idx}` instead)
        # since otherwise we have to emit them exactly in order or llvm rejects the program
        res = f"{prefix}.{idx}"

        if llvm_id_re.match(res) is not None:
            return res

        return llvmstr(res)

    def emit(self, x: str) -> None:
        self.lines.append("  " * self.indent_level + x)

    def compile_expr(self, x: Node) -> str:
        if isinstance(x, ExpressionList):
            return self.compile_expr(x.xs[0])

        if isinstance(x, ConditionalExpression):
            return self.compile_expr(x.then)

        if isinstance(x, StarredExpression):
            return self.compile_expr(x.x)

        if isinstance(x, OrTest):
            return self.compile_expr(x.rhs)

        if isinstance(x, AndTest):
            return self.compile_expr(x.rhs)

        if isinstance(x, NotTest):
            return self.compile_expr(x.x)

        if isinstance(x, Comparison):
            lhs = self.compile_expr(x.lhs)
            if x.ops is None:
                return lhs

            assert x.rhs is not None

            res = self.next_var(prefix="_compare")
            self.emit(f"%{res} = bitcast i1 1 to i1")

            cur = lhs
            for op, rhs_node in zip(x.ops, x.rhs, strict=True):
                rhs = self.compile_expr(rhs_node)

                prev_res = res
                res = self.next_var(prefix="_eq")

                assert op.text == "=="

                subres = self.next_var()
                self.emit(f"%{subres} = icmp eq i64 %{cur}, %{rhs}")
                self.emit(f"%{res} = and i1 %{prev_res}, %{subres}")

                cur = rhs

            final_res = self.next_var(prefix="_compare.res")
            self.emit(f"%{final_res} = zext i1 %{res} to i64")
            return final_res

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

            assert x.op is not None

            lhs = self.compile_expr(x.lhs)
            rhs = self.compile_expr(x.rhs)

            res = self.next_var()

            match x.op.text:
                case "+":
                    self.emit(f"%{res} = add i64 %{lhs}, %{rhs}")

                case _:
                    raise NotImplementedError(f"unsupported oeprator: {x.op.text}")

            return res

        if isinstance(x, MExpr):
            if x.lhs is None:
                return self.compile_expr(x.rhs)

            assert x.op is not None

            lhs = self.compile_expr(x.lhs)
            rhs = self.compile_expr(x.rhs)

            res = self.next_var()

            match x.op.text:
                case "%":
                    self.emit(f"%{res} = urem i64 %{lhs}, %{rhs}")

                case "*":
                    self.emit(f"%{res} = mul i64 %{lhs}, %{rhs}")

                case _:
                    raise NotImplementedError(f"unsupported oeprator: {x.op.text}")

            return res

        if isinstance(x, UExpr):
            return self.compile_expr(x.x)

        if isinstance(x, Power):
            return self.compile_expr(x.x)

        if isinstance(x, Primary):
            return self.compile_expr(x.x)

        if isinstance(x, Atom):
            if isinstance(x.x, Token):
                assert x.x.type == "identifier"
                return self.locals[x.x.nfkd()]

            return self.compile_expr(x.x)

        if isinstance(x, AstLiteral):
            res = self.next_var()

            match x.x.type:
                case "decinteger":
                    self.emit(f"%{res} = bitcast i64 {x.x.text} to i64")
                case "identifier":
                    match x.x.text:
                        case "True":
                            self.emit(f"%{res} = bitcast i64 1 to i64")
                        case "False":
                            self.emit(f"%{res} = bitcast i64 0 to i64")
                        case _:
                            raise NotImplementedError(f"unsupported named literal: {x.x.text}")
                case _:
                    raise NotImplementedError(f"unknown literal: {x}")
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
            self.emit("\n".join(f"; {l}" for l in x.unparse().splitlines()))
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

        if isinstance(x, AssignmentStmt):
            assert len(x.targets) == 1

            target = x.targets[0]
            assert len(target.xs) == 1

            name = target.xs[0].x.nfkd()
            var = self.next_var(prefix=name)
            self.locals[name] = var

            val = self.compile_expr(x.value)

            self.emit(f"%{var} = bitcast i64 %{val} to i64")
            self.emit("")
            return

        if isinstance(x, AugmentedAssignmentStmt):
            name = x.target.x.nfkd()

            val = self.compile_expr(x.value)

            old = self.locals[name]
            res = self.next_var(prefix=name)
            self.locals[name] = res

            match x.op.text:
                case "+=":
                    self.emit(f"%{res} = add i64 %{old}, %{val}")

                case _:
                    raise NotImplementedError(
                        f"unsupported augmented assignment operator: {x.op.text}"
                    )

            self.emit("")
            return

        raise NotImplementedError(f"unknown node: {x.type}")
