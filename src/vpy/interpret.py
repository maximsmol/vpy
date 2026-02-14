import struct
from dataclasses import dataclass, field
from typing import Any, Literal, Self, override

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


@dataclass(kw_only=True)
class Value:
    type_id: Literal["None", "int", "bool", "float", "function"]
    _value: bytes

    # from_*
    @classmethod
    def from_none(cls) -> Self:
        return cls(type_id="None", _value=b"")

    @classmethod
    def from_int(cls, x: int) -> Self:
        return cls(type_id="int", _value=x.to_bytes())

    @classmethod
    def from_bool(cls, x: bool) -> Self:  # noqa: FBT001
        return cls(type_id="bool", _value=bytes([1 if x else 0]))

    @classmethod
    def from_float(cls, x: float) -> Self:
        return cls(type_id="float", _value=struct.pack("d", x))

    @classmethod
    def from_function(cls, name: str) -> Self:
        return cls(type_id="function", _value=name.encode())

    # expect_*
    def expect_none(self) -> None:
        assert self.type_id == "None"

    def expect_int(self) -> int:
        assert self.type_id == "int"
        return int.from_bytes(self._value)

    def expect_bool(self) -> bool:
        assert self.type_id == "bool"
        return self._value[0] != 0

    def expect_float(self) -> float:
        assert self.type_id == "float"
        return struct.unpack("d", self._value)[0]

    def expect_function(self) -> str:
        assert self.type_id == "function"
        return self._value.decode()

    def to_python(self, interpreter: "Interpreter") -> object:
        match self.type_id:
            case "None":
                return self.expect_none()
            case "int":
                return self.expect_int()
            case "bool":
                return self.expect_bool()
            case "float":
                return self.expect_float()
            case "function":
                return interpreter.functions[self.expect_function()]
            case _:
                raise RuntimeError(
                    f"value cannot be converted into a native representation: type_id={self.type_id}"
                )


class FunctionReturn(Exception):  # noqa: N818
    def __init__(self, value: Value) -> None:
        super().__init__(self)
        self.value: Value = value


@dataclass(kw_only=True)
class Function:
    data: Funcdef

    @override
    def __repr__(self) -> str:
        return f"<function {self.data.name.nfkd()}>"


@dataclass(kw_only=True)
class Scope:
    interpreter: "Interpreter"

    locals: dict[str, Value] = field(default_factory=dict)

    def resolve(self, x: str) -> Value:
        if x not in self.locals:
            raise RuntimeError(f"variable not defined: {x}")

        return self.locals[x]

    def eval(self, x: Node) -> Value:
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
                            case "int":
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
                    return Value.from_bool(False)  # noqa: FBT003
                cur = rhs

            return Value.from_bool(True)  # noqa: FBT003

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
                    return Value.from_int(lhs.expect_int() + rhs.expect_int())

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
                    return Value.from_int(lhs.expect_int() % rhs.expect_int())

                case "*":
                    return Value.from_int(lhs.expect_int() * rhs.expect_int())

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
            func = self.interpreter.functions[func_val.expect_function()]

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

            return Value.from_none()

        if isinstance(x, Atom):
            if isinstance(x.x, Token):
                assert x.x.type == "identifier"
                return self.resolve(x.x.nfkd())

            return self.eval(x.x)

        if isinstance(x, AstLiteral):
            match x.x.type:
                case "decinteger":
                    return Value.from_int(int(x.x.text))

                case "floatnumber":
                    return Value.from_float(float(x.x.text))

                case "identifier":
                    match x.x.text:
                        case "True":
                            return Value.from_bool(True)  # noqa: FBT003
                        case "False":
                            return Value.from_bool(False)  # noqa: FBT003
                        case "None":
                            return Value.from_none()
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

            self.locals[name] = Value.from_function(f_id)
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
                    self.locals[name] = Value.from_int(
                        cur.expect_int() + val.expect_int()
                    )
                case _:
                    raise NotImplementedError(
                        f"unsupported augmented assignment operator: {x.op.text}"
                    )

            return

        if isinstance(x, ReturnStmt):
            value: Value = Value.from_none()
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
