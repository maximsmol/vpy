from dataclasses import dataclass, field
from typing import override

from .lex import Token
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
from .values import VpyTypeId, VpyValue


class FunctionReturn(Exception):  # noqa: N818
    def __init__(self, value: VpyValue) -> None:
        super().__init__(self)
        self.value: VpyValue = value


@dataclass(kw_only=True)
class Function:
    data: Funcdef

    @override
    def __repr__(self) -> str:
        return f"<function {self.data.name.nfkd()}>"


@dataclass(kw_only=True)
class Scope:
    interpreter: "Interpreter"

    locals: dict[str, VpyValue] = field(default_factory=dict)

    def resolve(self, x: str) -> VpyValue:
        if x not in self.locals:
            raise RuntimeError(f"variable not defined: {x}")

        return self.locals[x]

    def eval(self, x: Node) -> VpyValue:
        if isinstance(x, ExpressionList):
            return self.eval(x.xs[0])

        if isinstance(x, Expression):
            return self.eval(x.x)

        if isinstance(x, ConditionalExpression):
            return self.eval(x.then)

        if isinstance(x, AssignmentExpression):
            return self.eval(x.value)

        if isinstance(x, StarredExpression):
            return self.eval(x.x)

        if isinstance(x, OrTest):
            return self.eval(x.rhs)

        if isinstance(x, AndTest):
            return self.eval(x.rhs)

        if isinstance(x, NotTest):
            return self.eval(x.x)

        if isinstance(x, Comparison):
            lhs = self.eval(x.lhs)
            if x.ops is None:
                return lhs

            assert x.rhs is not None

            cur = lhs
            for op, rhs_node in zip(x.ops, x.rhs, strict=True):
                rhs = self.eval(rhs_node)

                match op.text:
                    case "==":
                        match cur.type_id:
                            case VpyTypeId.int:
                                cond = cur.expect_int() == rhs.expect_int()
                            case _:
                                raise RuntimeError(
                                    f"unsupported equality: {cur} == {rhs}"
                                )
                    case "<=":
                        cond = cur.expect_int() <= rhs.expect_int()
                    case _:
                        raise RuntimeError(
                            f"unsupported comparison operator: {op.text}"
                        )

                if not cond:
                    return VpyValue.from_bool(False)  # noqa: FBT003
                cur = rhs

            return VpyValue.from_bool(True)  # noqa: FBT003

        if isinstance(x, OrExpr):
            return self.eval(x.rhs)

        if isinstance(x, XorExpr):
            return self.eval(x.rhs)

        if isinstance(x, AndExpr):
            return self.eval(x.rhs)

        if isinstance(x, ShiftExpr):
            return self.eval(x.rhs)

        if isinstance(x, AExpr):
            if x.lhs is None:
                return self.eval(x.rhs)

            assert x.op is not None

            lhs = self.eval(x.lhs)
            rhs = self.eval(x.rhs)

            match x.op.text:
                case "+":
                    return VpyValue.from_int(lhs.expect_int() + rhs.expect_int())

                case _:
                    raise NotImplementedError(f"unsupported operator: {x.op.text}")

        if isinstance(x, MExpr):
            if x.lhs is None:
                return self.eval(x.rhs)

            assert x.op is not None

            lhs = self.eval(x.lhs)
            rhs = self.eval(x.rhs)

            match x.op.text:
                case "%":
                    return VpyValue.from_int(lhs.expect_int() % rhs.expect_int())

                case "*":
                    return VpyValue.from_int(lhs.expect_int() * rhs.expect_int())

                case _:
                    raise NotImplementedError(f"unsupported operator: {x.op.text}")

        if isinstance(x, UExpr):
            return self.eval(x.x)

        if isinstance(x, Power):
            return self.eval(x.x)

        if isinstance(x, Primary):
            return self.eval(x.x)

        if isinstance(x, Call):
            func_val = self.eval(x.func)
            func = self.interpreter.functions[func_val.interpreted_expect_function()]

            param_spec = func.data.params
            if len(x.positional_args) != len(param_spec.regular_params):
                raise RuntimeError("wrong parameter count")

            func_scope = Scope(interpreter=self.interpreter)
            for k, v in zip(param_spec.regular_params, x.positional_args, strict=True):
                func_scope.locals[k.name.nfkd()] = self.eval(v)

            try:
                func_scope.exec(func.data.body)
            except FunctionReturn as e:
                return e.value

            return VpyValue.from_none()

        if isinstance(x, Atom):
            if isinstance(x.x, Token):
                assert x.x.type == "identifier"
                return self.resolve(x.x.nfkd())

            return self.eval(x.x)

        if isinstance(x, AstLiteral):
            match x.x.type:
                case "decinteger":
                    return VpyValue.from_int(int(x.x.text))

                case "floatnumber":
                    return VpyValue.from_float(float(x.x.text))

                case "identifier":
                    match x.x.text:
                        case "True":
                            return VpyValue.from_bool(True)  # noqa: FBT003
                        case "False":
                            return VpyValue.from_bool(False)  # noqa: FBT003
                        case "None":
                            return VpyValue.from_none()
                        case _:
                            raise NotImplementedError(
                                f"unknown named literal: {x.x.text}"
                            )

                case _:
                    raise NotImplementedError(f"unknown literal token: {x.x}")

        raise NotImplementedError(f"unknown node: {x.type}")

    def exec(self, x: Node) -> None:
        if isinstance(x, FileInput):
            for s in x.xs:
                self.exec(s)
            return

        if isinstance(x, Statement):
            self.exec(x.x)
            return

        if isinstance(x, CompoundStmt):
            self.exec(x.x)
            return

        if isinstance(x, StmtList):
            for s in x.xs:
                self.exec(s)
            return

        if isinstance(x, Suite):
            if isinstance(x.xs, StmtList):
                self.exec(x.xs)
                return

            for s in x.xs:
                self.exec(s)
            return

        if isinstance(x, IfStmt):
            cond = self.eval(x.cond)

            if cond.expect_bool():
                self.exec(x.then)
            return

        if isinstance(x, WhileStmt):
            cond = self.eval(x.cond)

            while cond.expect_bool():
                self.exec(x.loop)
                cond = self.eval(x.cond)
            return

        if isinstance(x, Funcdef):
            idx = self.interpreter.next_function_idx()

            name = x.name.nfkd()
            f_id = f"{idx}.{name}"

            self.locals[name] = VpyValue.interpreted_from_function(f_id)
            self.interpreter.functions[f_id] = Function(data=x)

            return

        if isinstance(x, SimpleStmt):
            self.exec(x.x)
            return

        if isinstance(x, ExpressionStmt):
            _ = self.eval(x.x)
            return

        if isinstance(x, AssignmentStmt):
            assert len(x.targets) == 1

            target = x.targets[0]
            assert len(target.xs) == 1

            name = target.xs[0]

            val = self.eval(x.value)
            self.locals[name.x.nfkd()] = val

            return

        if isinstance(x, AugmentedAssignmentStmt):
            name = x.target.x.nfkd()
            cur = self.locals[name]

            val = self.eval(x.value)

            match x.op.text:
                case "+=":
                    self.locals[name] = VpyValue.from_int(
                        cur.expect_int() + val.expect_int()
                    )
                case _:
                    raise NotImplementedError(
                        f"unsupported augmented assignment operator: {x.op.text}"
                    )

            return

        if isinstance(x, ReturnStmt):
            value: VpyValue = VpyValue.from_none()
            if x.value is not None:
                value = self.eval(x.value)
            raise FunctionReturn(value)

        raise NotImplementedError(f"unknown node: {x.type}")


@dataclass(kw_only=True)
class Interpreter:
    functions: dict[str, Function] = field(default_factory=dict)
    function_idx: int = 0

    def next_function_idx(self) -> int:
        res = self.function_idx
        self.function_idx += 1

        return res
