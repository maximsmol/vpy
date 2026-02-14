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
    Call,
    Comparison,
    CompoundStmt,
    ConditionalExpression,
    Expression,
    ExpressionList,
    ExpressionStmt,
    FileInput,
    Funcdef,
    IfStmt,
    MExpr,
    Node,
    NotTest,
    OrExpr,
    OrTest,
    Power,
    Primary,
    ReturnStmt,
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


def llvmid(prefix: str, idx: int) -> str:
    # note: we deliberately do not use unnamed variables (and use `.{idx}` instead)
    # since otherwise we have to emit them exactly in order or llvm rejects the program
    res = f"{prefix}.{idx}"

    if llvm_id_re.match(res) is not None:
        return res

    return llvmstr(res)


@dataclass(kw_only=True)
class Scope:
    compiler: "Compiler"
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

        return llvmid(prefix, idx)

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

                cur_data = self.expect_int(cur)
                rhs_data = self.expect_int(rhs)

                match op.text:
                    case "==":
                        self.emit(f"%{subres} = icmp eq i64 %{cur_data}, %{rhs_data}")
                    case "<=":
                        self.emit(f"%{subres} = icmp sle i64 %{cur_data}, %{rhs_data}")
                    case _:
                        raise RuntimeError(
                            f"unsupported comparison operator: {op.text}"
                        )

                self.emit(f"%{res} = and i1 %{prev_res}, %{subres}")
                cur = rhs

            res_i8 = self.next_id(prefix="_compare.i8")
            self.emit(f"%{res_i8} = zext i1 %{res} to i8")
            return self.make_bool(res_i8)

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

            match x.op.text:
                case "+":
                    lhs_data = self.expect_int(lhs)
                    rhs_data = self.expect_int(rhs)

                    res_data = self.next_id(prefix="_aexpr.data")
                    self.emit(f"%{res_data} = add i64 %{lhs_data}, %{rhs_data}")

                    res = self.make_int(res_data)

                case _:
                    raise NotImplementedError(f"unsupported oeprator: {x.op.text}")

            return res

        if isinstance(x, MExpr):
            if x.lhs is None:
                return self.compile_expr(x.rhs)

            assert x.op is not None

            lhs = self.compile_expr(x.lhs)
            rhs = self.compile_expr(x.rhs)

            match x.op.text:
                case "%":
                    lhs_data = self.expect_int(lhs)
                    rhs_data = self.expect_int(rhs)

                    res_data = self.next_id(prefix="_mexpr.data")
                    self.emit(f"%{res_data} = urem i64 %{lhs_data}, %{rhs_data}")

                    res = self.make_int(res_data)

                case "*":
                    lhs_data = self.expect_int(lhs)
                    rhs_data = self.expect_int(rhs)

                    res_data = self.next_id(prefix="_mexpr.data")
                    self.emit(f"%{res_data} = mul i64 %{lhs_data}, %{rhs_data}")

                    res = self.make_int(res_data)
                case _:
                    raise NotImplementedError(f"unsupported oeprator: {x.op.text}")

            return res

        if isinstance(x, UExpr):
            return self.compile_expr(x.x)

        if isinstance(x, Power):
            return self.compile_expr(x.x)

        if isinstance(x, Primary):
            return self.compile_expr(x.x)

        if isinstance(x, Call):
            f = self.compile_expr(x.func)

            params = x.positional_args

            args = self.next_id(prefix="call.args")
            self.emit(f"%{args} = alloca ptr, i64 {len(params)}")

            for idx, param in enumerate(params):
                param_var = self.compile_expr(param)

                arg = self.next_id(prefix=f"call.args.{idx}")
                self.emit(f"%{arg} = getelementptr ptr, ptr %{args}, i64 {idx}")
                self.emit(f"store ptr %{param_var}, ptr %{arg}")

            self.emit("")

            res = self.next_id(prefix="call.res")
            self.emit(f"%{res} = call ptr %{f}(ptr %{args})")

            return res

        if isinstance(x, Atom):
            if isinstance(x.x, Token):
                assert x.x.type == "identifier"
                return self.locals[x.x.nfkd()]

            return self.compile_expr(x.x)

        if isinstance(x, AstLiteral):
            match x.x.type:
                case "decinteger":
                    res_data = self.next_id(prefix="_const.data")
                    self.emit(f"%{res_data} = bitcast i64 {x.x.text} to i64")

                    res = self.make_int(res_data)
                case "floatnumber":
                    res_data = self.next_id(prefix="_const.data")
                    self.emit(f"%{res_data} = bitcast double {x.x.text} to double")

                    res = self.make_float(res_data)
                case "identifier":
                    match x.x.text:
                        case "True":
                            res_data = self.next_id(prefix="_const.data")
                            self.emit(f"%{res_data} = bitcast i8 1 to i8")

                            res = self.make_bool(res_data)
                        case "False":
                            res_data = self.next_id(prefix="_const.data")
                            self.emit(f"%{res_data} = bitcast i8 0 to i8")

                            res = self.make_bool(res_data)
                        case "None":
                            res = self.make_none()
                        case _:
                            raise NotImplementedError(
                                f"unsupported named literal: {x.x.text}"
                            )
                case _:
                    raise NotImplementedError(f"unknown literal: {x}")
            return res

        raise NotImplementedError(f"unknown node: {x.type}")

    def compile_func_body(self, x: Funcdef, *, name: str) -> None:
        self.emit(f"define ptr @{name}(ptr %args) {{")
        with self.indent():
            self.emit("start:")

            params = x.params.regular_params
            for idx, param in enumerate(params):
                name = param.name.nfkd()

                var = self.next_id(prefix=name)
                arg = self.next_id(prefix=f"arg.ptr.{name}")

                self.emit(f"%{arg} = getelementptr ptr, ptr %args, i64 {idx}")
                self.emit(f"%{var} = load ptr, ptr %{arg}")

                self.locals[name] = var

            self.compile(x.body)

            self.emit("; fallback: return None")
            none = self.make_none()
            self.emit(f"ret ptr %{none}")
        self.emit("}")

    def compile(self, x: Node) -> None:
        if isinstance(x, FileInput):
            self.emit("declare void @abort() noreturn")
            self.emit("declare ptr @malloc(i64)")
            self.emit("declare i64 @puts(ptr)")
            self.emit("")
            self.emit(r'@str.error = constant [19 x i8] c"[panic] type error\00"')
            self.emit("")
            self.emit("%struct.value = type { i64, [ 0 x i8 ] }")
            self.emit("")
            self.emit("define ptr @module_root() {")
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
            loop_epilog_label = self.next_id(prefix="while.loop.epilog")

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
                    f"%{var} = phi ptr [ %{prev}, %{prev_block} ], [ %{new}, %{loop_epilog_label} ]"
                )
                self.locals[k] = var

            cond = self.compile_expr(x.cond)
            cond_data = self.expect_bool(cond)

            cond_i1 = self.next_id(prefix="while.cond.i1")
            self.emit(f"%{cond_i1} = icmp eq i8 %{cond_data}, 1")

            end_label = self.next_id(prefix="while.end")

            post_cond_locals = copy(self.locals)
            self.emit(f"br i1 %{cond_i1}, label %{loop_label}, label %{end_label}")

            # >>> while.loop
            self.emit_block(loop_label)
            self.compile(x.loop)

            self.emit(f"br label %{loop_epilog_label}")

            # >>> while.loop.epilog
            self.emit_block(loop_epilog_label)
            for k, new in loop_end_locals.items():
                self.emit(f"%{new} = bitcast ptr %{self.locals[k]} to ptr")

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

        if isinstance(x, Funcdef):
            name = x.name.nfkd()
            scope_id = self.compiler.next_scope_id(prefix=name)

            fn_scope = Scope(compiler=self.compiler)
            self.compiler.scopes.append(fn_scope)

            fn_scope.compile_func_body(x, name=scope_id)

            var = self.next_id(prefix=name)
            self.emit(f"%{var} = bitcast ptr @{scope_id} to ptr")
            self.locals[x.name.nfkd()] = var

            return

        if isinstance(x, IfStmt):
            prev_locals = copy(self.locals)

            cond = self.compile_expr(x.cond)
            cond_data = self.expect_bool(cond)

            cond_i1 = self.next_id(prefix="if.cond.i1")
            self.emit(f"%{cond_i1} = icmp eq i8 %{cond_data}, 1")

            then_label = self.next_id(prefix="if.then")
            then_epilog_label = self.next_id(prefix="if.then.epilog")
            end_label = self.next_id(prefix="if.end")

            cond_epilog_label = self.next_id(prefix="if.cond.epilog")
            self.emit(f"br label %{cond_epilog_label}")

            self.emit_block(cond_epilog_label)
            self.emit(f"br i1 %{cond_i1}, label %{then_label}, label %{end_label}")

            self.emit_block(then_label)
            self.compile(x.then)
            self.emit(f"br label %{then_epilog_label}")

            self.emit_block(then_epilog_label)
            self.emit(f"br label %{end_label}")

            self.emit_block(end_label)

            for k, prev in prev_locals.items():
                new = self.locals[k]
                if prev == new:
                    continue

                var = self.next_id(prefix=k)

                self.emit(
                    f"%{var} = phi ptr [ %{prev}, %{cond_epilog_label} ], [ %{new}, %{then_epilog_label} ]"
                )
                self.locals[k] = var

            self.emit("")

            return

        if isinstance(x, SimpleStmt):
            self.compile(x.x)
            return

        if isinstance(x, ExpressionStmt):
            res = self.compile_expr(x.x)
            self.emit(f"ret ptr %{res}")
            return

        if isinstance(x, AssignmentStmt):
            assert len(x.targets) == 1

            target = x.targets[0]
            assert len(target.xs) == 1

            name = target.xs[0].x.nfkd()
            var = self.next_id(prefix=name)

            val = self.compile_expr(x.value)

            self.emit(f"%{var} = bitcast ptr %{val} to ptr")
            self.emit("")
            self.locals[name] = var

            return

        if isinstance(x, AugmentedAssignmentStmt):
            name = x.target.x.nfkd()

            val = self.compile_expr(x.value)

            old = self.locals[name]

            match x.op.text:
                case "+=":
                    old_data = self.expect_int(old)
                    val_data = self.expect_int(val)

                    res_data = self.next_id(prefix=f"{name}.data")
                    self.emit(f"%{res_data} = add i64 %{old_data}, %{val_data}")

                    res = self.make_int(res_data)

                case _:
                    raise NotImplementedError(
                        f"unsupported augmented assignment operator: {x.op.text}"
                    )

            self.locals[name] = res
            self.emit("")
            return

        if isinstance(x, ReturnStmt):
            if x.value is None:
                self.emit("ret void")
                return

            val = self.compile_expr(x.value)
            self.emit(f"ret ptr %{val}")
            return

        raise NotImplementedError(f"unknown node: {x.type}")

    def struct_sizeof(self, x: str) -> str:
        size_ptr = self.next_id(prefix="struct_sizeof.size_ptr")
        self.emit(f"%{size_ptr} = getelementptr {x}, ptr null, i64 1")

        res = self.next_id(prefix="struct_sizeof.res")
        self.emit(f"%{res} = ptrtoint ptr %{size_ptr} to i64")

        return res

    def make_none(self) -> str:
        res = self.next_id(prefix="make_none.res")
        self.emit(f"%{res} = call ptr @malloc(i64 %{self.struct_sizeof('{i64}')})")

        self.emit(f"store i64 0, ptr %{res}")

        return res

    def make_int(self, x: str) -> str:

        res = self.next_id(prefix="make_int.res")
        self.emit(f"%{res} = call ptr @malloc(i64 %{self.struct_sizeof('{i64, i64}')})")

        self.emit(f"store i64 1, ptr %{res}")

        data_ptr = self.next_id(prefix="make_int.data_ptr")
        self.emit(
            f"%{data_ptr} = getelementptr %struct.value, ptr %{res}, i64 0, i32 1"
        )
        self.emit(f"store i64 %{x}, ptr %{data_ptr}")

        return res

    def make_bool(self, x: str) -> str:
        res = self.next_id(prefix="make_bool.res")
        self.emit(f"%{res} = call ptr @malloc(i64 %{self.struct_sizeof('{i64, i8}')})")

        self.emit(f"store i64 2, ptr %{res}")

        data_ptr = self.next_id(prefix="make_bool.data_ptr")
        self.emit(
            f"%{data_ptr} = getelementptr %struct.value, ptr %{res}, i64 0, i32 1"
        )
        self.emit(f"store i8 %{x}, ptr %{data_ptr}")

        return res

    def make_float(self, x: str) -> str:
        res = self.next_id(prefix="make_float.res")
        self.emit(
            f"%{res} = call ptr @malloc(i64 %{self.struct_sizeof('{i64, double}')})"
        )

        self.emit(f"store i64 3, ptr %{res}")

        data_ptr = self.next_id(prefix="make_float.data_ptr")
        self.emit(
            f"%{data_ptr} = getelementptr %struct.value, ptr %{res}, i64 0, i32 1"
        )
        self.emit(f"store double %{x}, ptr %{data_ptr}")

        return res

    def expect_type(self, x: str, expected: int) -> None:
        type_id = self.next_id(prefix="expect_int.type_id")
        self.emit(f"%{type_id} = load i64, ptr %{x}")

        cond_i1 = self.next_id(prefix="expect_type.cond_i1")
        self.emit(f"%{cond_i1} = icmp eq i64 %{type_id}, {expected}")

        ok_label = self.next_id(prefix="expect_type.ok")
        crash_label = self.next_id(prefix="expect_type.crash")

        self.emit(f"br i1 %{cond_i1}, label %{ok_label}, label %{crash_label}")
        self.emit_block(crash_label)

        self.emit("call i64 @puts(ptr @str.error)")
        self.emit("call void @abort()")
        self.emit(f"br label %{ok_label}")

        self.emit_block(ok_label)

    def expect_int(self, x: str) -> str:
        self.expect_type(x, 1)

        # todo(maximsmol): throw on wrong type
        ptr = self.next_id(prefix="expect_int.ptr")
        self.emit(f"%{ptr} = getelementptr %struct.value, ptr %{x}, i64 0, i32 1")

        res = self.next_id(prefix="expect_int.res")
        self.emit(f"%{res} = load i64, ptr %{ptr}")

        return res

    def expect_bool(self, x: str) -> str:
        self.expect_type(x, 2)

        # todo(maximsmol): throw on wrong type
        ptr = self.next_id(prefix="expect_bool.ptr")
        self.emit(f"%{ptr} = getelementptr %struct.value, ptr %{x}, i64 0, i32 1")

        res = self.next_id(prefix="expect_bool.res")
        self.emit(f"%{res} = load i8, ptr %{ptr}")

        return res

    def expect_float(self, x: str) -> str:
        self.expect_type(x, 3)

        # todo(maximsmol): throw on wrong type
        ptr = self.next_id(prefix="expect_float.ptr")
        self.emit(f"%{ptr} = getelementptr %struct.value, ptr %{x}, i64 0, i32 1")

        res = self.next_id(prefix="expect_float.res")
        self.emit(f"%{res} = load double, ptr %{ptr}")

        return res


@dataclass(kw_only=True)
class Compiler:
    scopes: list[Scope] = field(default_factory=list)
    scope_idx: int = 0

    def next_scope_id(self, *, prefix: str = "") -> str:
        idx = self.scope_idx
        self.scope_idx += 1

        return llvmid(prefix, idx)
