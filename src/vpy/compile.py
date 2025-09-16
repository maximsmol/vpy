import re
from collections.abc import Generator
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, field

from vpy.lex import Token

from .parse import (
    AExpr,
    AndExpr,
    AndTest,
    AssignmentExpression,
    AssignmentStmt,
    AstLiteral,
    Atom,
    AugmentedAssignmentStmt,
    Comparison,
    CompoundStmt,
    ConditionalExpression,
    Expression,
    ExpressionList,
    ExpressionStmt,
    FileInput,
    IfStmt,
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
    Suite,
    UExpr,
    WhileStmt,
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
    cur_block: str = "start"

    locals: dict[str, str] = field(default_factory=dict)

    @contextmanager
    def indent(self) -> Generator[None]:
        try:
            self.indent_level += 1
            yield
        finally:
            self.indent_level -= 1

    def next_id(self, *, prefix: str = "") -> str:
        idx = self.var_idx
        self.var_idx += 1

        # note: we deliberately do not use unnamed variables (and use `.{idx}` instead)
        # since otherwise we have to emit them exactly in order or llvm rejects the program
        res = f"{prefix}.{idx}"

        if llvm_id_re.match(res) is not None:
            return res

        return llvmstr(res)

    def emit_block(self, name: str) -> None:
        self.emit("")
        self.emit(f"{name}:")
        self.cur_block = name

    def emit(self, x: str) -> None:
        self.lines.append("  " * self.indent_level + x)

    def emit_unparse(self, x: Node) -> None:
        self.lines.extend(
            "  " * self.indent_level + f"; {l}" for l in x.unparse().splitlines()
        )

    def compile_expr(self, x: Node) -> str:
        if isinstance(x, ExpressionList):
            return self.compile_expr(x.xs[0])

        if isinstance(x, Expression):
            self.emit_unparse(x)
            return self.compile_expr(x.x)

        if isinstance(x, ConditionalExpression):
            return self.compile_expr(x.then)

        if isinstance(x, AssignmentExpression):
            return self.compile_expr(x.value)

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

            res = self.next_id(prefix="_compare.i1")
            self.emit(f"%{res} = bitcast i1 1 to i1")

            cur = lhs
            for op, rhs_node in zip(x.ops, x.rhs, strict=True):
                rhs = self.compile_expr(rhs_node)

                prev_res = res
                res = self.next_id(prefix="_eq")

                subres = self.next_id(prefix="_compare.subres")
                match op.text:
                    case "==":
                        self.emit(f"%{subres} = icmp eq i64 %{cur}, %{rhs}")
                    case "<=":
                        self.emit(f"%{subres} = icmp sle i64 %{cur}, %{rhs}")
                    case _:
                        raise RuntimeError(
                            f"unsupported comparison operator: {op.text}"
                        )

                self.emit(f"%{res} = and i1 %{prev_res}, %{subres}")
                cur = rhs

            final_res = self.next_id(prefix="_compare.i64")
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

            res = self.next_id(prefix="_aexpr")

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

            res = self.next_id(prefix="_mexpr")

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
            res = self.next_id(prefix="_const")

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
                            raise NotImplementedError(
                                f"unsupported named literal: {x.x.text}"
                            )
                case _:
                    raise NotImplementedError(f"unknown literal: {x}")
            return res

        raise NotImplementedError(f"unknown node: {x.type}")

    def compile(self, x: Node) -> None:
        if isinstance(x, FileInput):
            self.emit("define i64 @test() {")
            with self.indent():
                self.emit("start:")
                for s in x.xs:
                    self.compile(s)
            self.emit("}")
            return

        if isinstance(x, Statement):
            self.emit_unparse(x)
            self.compile(x.x)
            return

        if isinstance(x, CompoundStmt):
            self.compile(x.x)
            return

        if isinstance(x, StmtList):
            for s in x.xs:
                self.compile(s)
            return

        if isinstance(x, Suite):
            if isinstance(x.xs, StmtList):
                self.compile(x.xs)
                return

            for s in x.xs:
                self.compile(s)
            return

        if isinstance(x, WhileStmt):
            prev_block = self.cur_block
            prev_locals = copy(self.locals)

            loop_cond_label = self.next_id(prefix="while.cond")
            loop_label = self.next_id(prefix="while.loop")

            self.emit(f"br label %{loop_cond_label}")

            lookahead_compiler = deepcopy(self)
            lookahead_compiler.compile(x.loop)
            lookahead_locals = lookahead_compiler.locals
            del lookahead_compiler

            loop_modified_locals: set[str] = set()
            for k, prev in prev_locals.items():
                la = lookahead_locals[k]
                if prev == la:
                    continue

                loop_modified_locals.add(k)

            loop_end_locals: dict[str, str] = {}
            for k in loop_modified_locals:
                loop_end_locals[k] = self.next_id(prefix=f"while.local.{k}")

            # >>> while.cond
            self.emit_block(loop_cond_label)
            for k in loop_modified_locals:
                var = self.next_id(prefix=k)

                prev = prev_locals[k]
                new = loop_end_locals[k]

                self.emit(
                    f"%{var} = phi i64 [ %{prev}, %{prev_block} ], [ %{new}, %{loop_label} ]"
                )
                self.locals[k] = var

            cond = self.compile_expr(x.cond)
            cond_i1 = self.next_id(prefix="while.cond.i1")
            self.emit(f"%{cond_i1} = icmp eq i64 %{cond}, 1")

            end_label = self.next_id(prefix="while.end")

            post_cond_locals = copy(self.locals)
            self.emit(f"br i1 %{cond_i1}, label %{loop_label}, label %{end_label}")

            # >>> while.loop
            self.emit_block(loop_label)
            self.compile(x.loop)

            for k, new in loop_end_locals.items():
                self.emit(f"%{new} = bitcast i64 %{self.locals[k]} to i64")

            self.emit(f"br label %{loop_cond_label}")

            # >>> while.end
            self.emit_block(end_label)
            self.emit("")
            for k, prev in post_cond_locals.items():
                new = self.locals[k]
                if prev == new:
                    continue

                self.locals[k] = prev

            return

        if isinstance(x, IfStmt):
            prev_block = self.cur_block
            prev_locals = copy(self.locals)

            cond = self.compile_expr(x.cond)
            cond_i1 = self.next_id(prefix="if.cond.i1")
            self.emit(f"%{cond_i1} = icmp eq i64 %{cond}, 1")

            then_label = self.next_id(prefix="if.then")
            end_label = self.next_id(prefix="if.end")

            self.emit(f"br i1 %{cond_i1}, label %{then_label}, label %{end_label}")

            self.emit_block(then_label)
            self.compile(x.then)
            self.emit(f"br label %{end_label}")

            self.emit_block(end_label)

            for k, prev in prev_locals.items():
                new = self.locals[k]
                if prev == new:
                    continue

                var = self.next_id(prefix=k)

                self.emit(
                    f"%{var} = phi i64 [ %{prev}, %{prev_block} ], [ %{new}, %{then_label} ]"
                )
                self.locals[k] = var

            self.emit("")

            return

        if isinstance(x, SimpleStmt):
            self.compile(x.x)
            return

        if isinstance(x, ExpressionStmt):
            res = self.compile_expr(x.x)
            self.emit(f"ret i64 %{res}")
            return

        if isinstance(x, AssignmentStmt):
            assert len(x.targets) == 1

            target = x.targets[0]
            assert len(target.xs) == 1

            name = target.xs[0].x.nfkd()
            var = self.next_id(prefix=name)

            val = self.compile_expr(x.value)

            self.emit(f"%{var} = bitcast i64 %{val} to i64")
            self.emit("")
            self.locals[name] = var

            return

        if isinstance(x, AugmentedAssignmentStmt):
            name = x.target.x.nfkd()

            val = self.compile_expr(x.value)

            old = self.locals[name]
            res = self.next_id(prefix=name)

            match x.op.text:
                case "+=":
                    self.emit(f"%{res} = add i64 %{old}, %{val}")

                case _:
                    raise NotImplementedError(
                        f"unsupported augmented assignment operator: {x.op.text}"
                    )

            self.locals[name] = res
            self.emit("")
            return

        raise NotImplementedError(f"unknown node: {x.type}")
