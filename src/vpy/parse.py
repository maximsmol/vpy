import abc
import ast
from abc import abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import wraps
from typing import Concatenate, override

from vpy.lex import Lexer, Token


@dataclass(kw_only=True)
class Node(abc.ABC):
    children: "list[Node | Token]" = field(default_factory=list)

    @property
    @abstractmethod
    def type(self) -> str:
        raise NotImplementedError

    def to_str(self, *, indent: str = "", visited: set[int] | None = None) -> str:
        if visited is None:
            visited = set()

        if id(self) in visited:
            return f"<recursion: {self.__class__.__name__}#{id(self)}>"
        visited.add(id(self))

        node_fields: dict[int, str] = {}

        fs = fields(self)
        for f in fs:
            if f.name in {"type", "children"}:
                continue

            val = getattr(self, f.name)
            if isinstance(val, list):
                for x in val:
                    if not isinstance(x, Node | Token):
                        continue

                    node_fields[id(x)] = f.name

                continue

            if not isinstance(val, Node | Token):
                continue

            node_fields[id(val)] = f.name

        children_lines: list[str] = []
        for x in self.children:
            field_name = node_fields.get(id(x))
            if field_name is None:
                if isinstance(x, Node):
                    children_lines.append(
                        f"{indent}  {x.to_str(indent=f'{indent}  ', visited=visited)}"
                    )
                else:
                    children_lines.append(f"{indent}  {x}")
            else:
                if isinstance(x, Node):
                    x_str = x.to_str(indent=f"{indent}  ", visited=visited)
                else:
                    x_str = str(x)

                children_lines.extend([f"{indent}  .{field_name}: " + x_str])

        lines: list[str] = [f"{self.type}#{id(self)}{{", *children_lines]
        lines.append(f"{indent}}}")

        return "\n".join(lines)

    @override
    def __str__(self) -> str:
        return self.to_str()

    @override
    def __repr__(self) -> str:
        return self.to_str()

    def unparse(self, *, parens: bool = False) -> str:
        res: list[str] = []

        for x in self.children:
            if isinstance(x, Token):
                res.append(x.text)
                continue

            res.append(x.unparse(parens=parens))

        return "".join(res)


@dataclass(kw_only=True)
class AstLiteral(Node):
    x: Token

    @property
    @override
    def type(self) -> str:
        return "literal"

    def to_ast(self) -> ast.Constant:
        match self.x.type:
            case "decinteger":
                return ast.Constant(value=int(self.x.text))

            case "identifier":
                match self.x.text:
                    case "True":
                        value = True
                    case "False":
                        value = False
                    case "None":
                        value = None
                    case _:
                        raise NotImplementedError(
                            f"unknown named literal: {self.x.text}"
                        )

                return ast.Constant(value=value)

            case _:
                raise NotImplementedError(f"unknown literal token: {self.x}")


@dataclass(kw_only=True)
class Atom(Node):
    x: AstLiteral | Token

    @property
    @override
    def type(self) -> str:
        return "atom"

    def to_ast(self) -> ast.Constant | ast.Name:
        if isinstance(self.x, Token):
            assert self.x.type == "identifier"
            return ast.Name(self.x.nfkd())

        return self.x.to_ast()


@dataclass(kw_only=True)
class Call(Node):
    func: "Primary"
    positional_args: "list[AssignmentExpression]"

    @property
    @override
    def type(self) -> str:
        return "call"

    def to_ast(self) -> ast.Call:
        return ast.Call(
            func=self.func.to_ast(), args=[x.to_ast() for x in self.positional_args]
        )


@dataclass(kw_only=True)
class Primary(Node):
    x: Atom | Call

    @property
    @override
    def type(self) -> str:
        return "primary"

    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class Power(Node):
    x: Primary

    @property
    @override
    def type(self) -> str:
        return "power"

    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class UExpr(Node):
    x: Power

    @property
    @override
    def type(self) -> str:
        return "u_expr"

    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class MExpr(Node):
    lhs: "MExpr | None"
    op: Token | None
    rhs: UExpr

    @property
    @override
    def type(self) -> str:
        return "m_expr"

    def to_ast(self) -> ast.BinOp | ast.expr:
        if self.lhs is None:
            return self.rhs.to_ast()

        # todo(maximsmol): use inheritance to make this obvious
        assert self.op is not None

        left = self.lhs.to_ast()
        assert isinstance(left, ast.expr)

        match self.op.text:
            case "*":
                op = ast.Mult()

            case "%":
                op = ast.Mod()

            case _:
                raise NotImplementedError(f"unknown operator: {self.op.text!r}")

        right = self.rhs.to_ast()
        assert isinstance(right, ast.expr)

        return ast.BinOp(left=left, op=op, right=right)

    @override
    def unparse(self, *, parens: bool = False) -> str:
        x = super().unparse(parens=parens)
        if parens:
            return f"({x})"

        return x


@dataclass(kw_only=True)
class AExpr(Node):
    lhs: "AExpr | None"
    op: Token | None
    rhs: MExpr

    @property
    @override
    def type(self) -> str:
        return "a_expr"

    def to_ast(self) -> ast.BinOp | ast.expr:
        if self.lhs is None:
            return self.rhs.to_ast()

        # todo(maximsmol): use inheritance to make this obvious
        assert self.op is not None

        left = self.lhs.to_ast()
        assert isinstance(left, ast.expr)

        match self.op.text:
            case "+":
                op = ast.Add()

            case _:
                raise NotImplementedError(f"unknown operator: {self.op.text!r}")

        right = self.rhs.to_ast()
        assert isinstance(right, ast.expr)

        return ast.BinOp(left=left, op=op, right=right)

    @override
    def unparse(self, *, parens: bool = False) -> str:
        x = super().unparse(parens=parens)
        if parens:
            return f"({x})"

        return x


@dataclass(kw_only=True)
class ShiftExpr(Node):
    rhs: AExpr

    @property
    @override
    def type(self) -> str:
        return "shift_expr"

    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class AndExpr(Node):
    rhs: ShiftExpr

    @property
    @override
    def type(self) -> str:
        return "and_expr"

    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class XorExpr(Node):
    rhs: AndExpr

    @property
    @override
    def type(self) -> str:
        return "xor_expr"

    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class OrExpr(Node):
    rhs: XorExpr

    @property
    @override
    def type(self) -> str:
        return "or_expr"

    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class Comparison(Node):
    lhs: OrExpr
    ops: list[Token] | None
    rhs: list[OrExpr] | None

    @property
    @override
    def type(self) -> str:
        return "comparison"

    def to_ast(self) -> ast.expr:
        if self.ops is None:
            return self.lhs.to_ast()

        # todo(maximsmol): use inheritance to make this obvious
        assert self.rhs is not None

        left = self.lhs.to_ast()

        comparators: list[ast.expr] = []
        for x in self.rhs:
            cur = x.to_ast()
            comparators.append(cur)

        def convert_op(x: Token) -> ast.Eq | ast.LtE:
            match x.text:
                case "==":
                    return ast.Eq()
                case "<=":
                    return ast.LtE()
                case _:
                    raise RuntimeError(f"unsupported comparison operator: {x.text}")

        return ast.Compare(
            left=left, ops=[convert_op(x) for x in self.ops], comparators=comparators
        )


@dataclass(kw_only=True)
class NotTest(Node):
    x: Comparison

    @property
    @override
    def type(self) -> str:
        return "note_test"

    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class AndTest(Node):
    rhs: NotTest

    @property
    @override
    def type(self) -> str:
        return "and_test"

    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class OrTest(Node):
    rhs: AndTest

    @property
    @override
    def type(self) -> str:
        return "or_test"

    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class StarredExpression(Node):
    x: OrTest

    @property
    @override
    def type(self) -> str:
        return "starred_expression"

    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class ExpressionStmt(Node):
    x: StarredExpression

    @property
    @override
    def type(self) -> str:
        return "expression_stmt"

    def to_ast(self) -> ast.Expr:
        return ast.Expr(value=self.x.to_ast())


@dataclass(kw_only=True)
class Target(Node):
    x: Token

    @property
    @override
    def type(self) -> str:
        return "target"

    def to_ast(self) -> ast.Name:
        return ast.Name(id=self.x.nfkd(), ctx=ast.Store())


@dataclass(kw_only=True)
class TargetList(Node):
    xs: list[Target]

    @property
    @override
    def type(self) -> str:
        return "target_list"

    def to_ast(self) -> ast.Name:
        assert len(self.xs) == 1
        return self.xs[0].to_ast()


@dataclass(kw_only=True)
class AssignmentStmt(Node):
    targets: list[TargetList]
    value: StarredExpression

    @property
    @override
    def type(self) -> str:
        return "assignment_stmt"

    def to_ast(self) -> ast.Assign:
        targets: list[ast.expr] = []
        for x in self.targets:
            cur = x.to_ast()
            assert isinstance(cur, ast.expr)
            targets.append(cur)

        value = self.value.to_ast()
        assert isinstance(value, ast.expr)

        return ast.Assign(targets=targets, value=value)


@dataclass(kw_only=True)
class AugTarget(Node):
    x: Token

    @property
    @override
    def type(self) -> str:
        return "augtarget"

    def to_ast(self) -> ast.Name:
        return ast.Name(id=self.x.nfkd(), ctx=ast.Store())


@dataclass(kw_only=True)
class ConditionalExpression(Node):
    then: OrTest

    @property
    @override
    def type(self) -> str:
        return "conditional_expression"

    def to_ast(self) -> ast.expr:
        return self.then.to_ast()


@dataclass(kw_only=True)
class AssignmentExpression(Node):
    value: "Expression"

    @property
    @override
    def type(self) -> str:
        return "assignment_expression"

    def to_ast(self) -> ast.expr:
        return self.value.to_ast()


@dataclass(kw_only=True)
class Expression(Node):
    x: ConditionalExpression

    @property
    @override
    def type(self) -> str:
        return "expression"

    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class ExpressionList(Node):
    xs: list[Expression]

    @property
    @override
    def type(self) -> str:
        return "expression_list"

    def to_ast(self) -> ast.expr:
        return self.xs[0].to_ast()


@dataclass(kw_only=True)
class AugmentedAssignmentStmt(Node):
    target: AugTarget
    op: Token
    value: ExpressionList

    @property
    @override
    def type(self) -> str:
        return "augmented_assignment_stmt"

    def to_ast(self) -> ast.AugAssign:
        target = self.target.to_ast()

        match self.op.text:
            case "+=":
                op = ast.Add()

            case _:
                raise NotImplementedError(
                    f"unknown augmented assignment operator: {self.op.text}"
                )

        value = self.value.to_ast()
        assert isinstance(value, ast.expr)

        return ast.AugAssign(target=target, op=op, value=value)


@dataclass(kw_only=True)
class ReturnStmt(Node):
    value: ExpressionList | None

    @property
    @override
    def type(self) -> str:
        return "return_stmt"

    def to_ast(self) -> ast.Return:
        return ast.Return(value=self.value.to_ast() if self.value is not None else None)


@dataclass(kw_only=True)
class SimpleStmt(Node):
    x: ExpressionStmt | AssignmentStmt | AugmentedAssignmentStmt | ReturnStmt

    @property
    @override
    def type(self) -> str:
        return "simple_stmt"

    def to_ast(self) -> ast.stmt:
        return self.x.to_ast()


@dataclass(kw_only=True)
class StmtList(Node):
    xs: list[SimpleStmt]

    @property
    @override
    def type(self) -> str:
        return "stmt_list"

    def to_ast(self) -> list[ast.stmt]:
        return [x.to_ast() for x in self.xs]


@dataclass(kw_only=True)
class IfStmt(Node):
    cond: AssignmentExpression
    then: "Suite"

    @property
    @override
    def type(self) -> str:
        return "if_stmt"

    def to_ast(self) -> ast.If:
        return ast.If(test=self.cond.to_ast(), body=self.then.to_ast())


@dataclass(kw_only=True)
class WhileStmt(Node):
    cond: AssignmentExpression
    loop: "Suite"

    @property
    @override
    def type(self) -> str:
        return "while_stmt"

    def to_ast(self) -> ast.While:
        return ast.While(test=self.cond.to_ast(), body=self.loop.to_ast())


@dataclass(kw_only=True)
class Defparameter(Node):
    name: Token
    annotation: Expression | None

    @property
    @override
    def type(self) -> str:
        return "defparameter"

    def to_ast(self) -> ast.arg:
        return ast.arg(
            arg=self.name.nfkd(),
            annotation=self.annotation.to_ast()
            if self.annotation is not None
            else None,
        )


@dataclass(kw_only=True)
class ParameterList(Node):
    regular_params: list[Defparameter]

    @property
    @override
    def type(self) -> str:
        return "parameter_list"

    def to_ast(self) -> ast.arguments:
        return ast.arguments(args=[x.to_ast() for x in self.regular_params])


@dataclass(kw_only=True)
class Funcdef(Node):
    name: Token
    params: ParameterList
    return_value: Expression | None
    body: "Suite"

    @property
    @override
    def type(self) -> str:
        return "funcdef"

    def to_ast(self) -> ast.FunctionDef:
        return ast.FunctionDef(
            name=self.name.nfkd(),
            args=self.params.to_ast(),
            body=self.body.to_ast(),
            returns=self.return_value.to_ast()
            if self.return_value is not None
            else None,
        )


@dataclass(kw_only=True)
class CompoundStmt(Node):
    x: IfStmt | WhileStmt | Funcdef

    @property
    @override
    def type(self) -> str:
        return "compound_stmt"

    def to_ast(self) -> ast.stmt:
        return self.x.to_ast()


@dataclass(kw_only=True)
class Statement(Node):
    x: StmtList | CompoundStmt

    @property
    @override
    def type(self) -> str:
        return "statement"

    def to_ast(self) -> list[ast.stmt]:
        if isinstance(self.x, CompoundStmt):
            return [self.x.to_ast()]

        return self.x.to_ast()


@dataclass(kw_only=True)
class Suite(Node):
    xs: StmtList | list[Statement]

    @property
    @override
    def type(self) -> str:
        return "suite"

    def to_ast(self) -> list[ast.stmt]:
        if isinstance(self.xs, StmtList):
            return self.xs.to_ast()

        res: list[ast.stmt] = []
        for x in self.xs:
            res.extend(x.to_ast())
        return res


@dataclass(kw_only=True)
class FileInput(Node):
    xs: list[Statement]

    @property
    @override
    def type(self) -> str:
        return "file_input"

    def to_ast(self) -> ast.Module:
        body: list[ast.stmt] = []

        for x in self.xs:
            body.extend(x.to_ast())

        return ast.Module(body=body)


type ParseFunctionIn[**T, R] = Callable[Concatenate[Parser, T], R]
type ParseFunctionOut[**T, R] = Callable[Concatenate[Parser, T], R]


def parse_function[**T, R: Node](f: ParseFunctionIn[T, R]) -> ParseFunctionOut[T, R]:
    @wraps(f)
    def res(self: "Parser", *args: T.args, **kwargs: T.kwargs) -> R:
        siblings = self.children
        try:
            self.children = []
            res = f(self, *args, **kwargs)
            if len(res.children) == 0:
                res.children = self.children

            siblings.append(res)

            return res
        except Exception as e:
            e.add_note(f"Parse function: {f.__name__}")
            raise
        finally:
            self.children = siblings

    return res


class ParseFailedError(RuntimeError): ...


class Parser:
    def __init__(self, *, lex: Lexer) -> None:
        self.lex: Lexer = lex
        self.children: list[Node | Token] = []

    @contextmanager
    def checkpoint(self) -> Generator[None]:
        old = deepcopy(self)

        try:
            yield
        except Exception:
            self.lex = old.lex
            self.children = old.children

            raise

    def tok(self) -> Token:
        res = self.lex.next()

        self.children.append(res)
        return res

    # todo(maximsmol): use token classes here instead of raw string type
    def opt(self, typ: str) -> Token | None:
        try:
            with self.checkpoint():
                x = self.tok()
                if x.type != typ:
                    raise ParseFailedError("opt")

                return x
        except ParseFailedError:
            return None

    def expect(self, typ: str) -> Token:
        res = self.tok()
        if res.type != typ:
            raise ParseFailedError(f"expected <{typ}>, found <{res.type}> ({res})")

        return res

    def expect_operator(self, text: str) -> Token:
        res = self.tok()
        if res.type != "operator":
            raise ParseFailedError(
                f"expected <operator {text}>, found <{res.type}> ({res})"
            )

        if res.text != text:
            raise ParseFailedError(
                f"expected <operator {text}>, found <operator {res.text}> ({res})"
            )

        return res

    def expect_delimiter(self, text: str) -> Token:
        res = self.tok()
        if res.type != "delimiter":
            raise ParseFailedError(
                f"expected <delimiter {text}>, found <{res.type}> ({res})"
            )

        if res.text != text:
            raise ParseFailedError(
                f"expected <delimiter {text}>, found <delimiter {res.text}> ({res})"
            )

        return res

    def expect_identifier(self, text: str) -> Token:
        res = self.tok()
        if res.type != "identifier":
            raise ParseFailedError(
                f"expected <identifier {text}>, found <{res.type}> ({res})"
            )

        if res.text != text:
            raise ParseFailedError(
                f"expected <identifier {text}>, found <identifier {res.text}> ({res})"
            )

        return res

    @parse_function
    def literal(self) -> AstLiteral:
        return AstLiteral(x=self.expect("decinteger"))

    @parse_function
    def atom(self) -> Atom:
        ident = self.opt("identifier")
        if ident is not None:
            if ident.text in {"True", "False", "None"}:
                return Atom(x=AstLiteral(x=ident))

            return Atom(x=ident)

        return Atom(x=self.literal())

    @parse_function
    def primary(self) -> Primary:
        res = Primary(x=self.atom())

        while True:
            try:
                with self.checkpoint():
                    _ = self.opt("whitespace")
                    _ = self.expect_delimiter("(")
                    _ = self.opt("whitespace")

                    positional_args: list[AssignmentExpression] = []
                    try:
                        positional_args.append(self.assignment_expression())

                        while True:
                            try:
                                with self.checkpoint():
                                    _ = self.opt("whitespace")
                                    _ = self.expect_delimiter(",")
                                    _ = self.opt("whitespace")
                                    positional_args.append(self.assignment_expression())
                            except ParseFailedError:
                                break
                    except ParseFailedError:
                        pass

                    _ = self.expect_delimiter(")")

                    res = Primary(x=Call(func=res, positional_args=positional_args))
            except ParseFailedError:
                break

        return res

    @parse_function
    def power(self) -> Power:
        return Power(x=self.primary())

    @parse_function
    def u_expr(self) -> UExpr:
        return UExpr(x=self.power())

    @parse_function
    def m_expr_base(self) -> MExpr:
        return MExpr(lhs=None, op=None, rhs=self.u_expr())

    @parse_function
    def m_expr(self) -> MExpr:
        lhs = self.m_expr_base()

        while True:
            try:
                with self.checkpoint():
                    _ = self.opt("whitespace")

                    op = self.expect("operator")
                    if op.text not in {"%", "*"}:
                        raise ParseFailedError("expected a `m_expr` operator")
            except ParseFailedError:
                return lhs

            _ = self.opt("whitespace")

            rhs = self.u_expr()

            lhs = MExpr(lhs=lhs, op=op, rhs=rhs)
            lhs.children = self.children

            self.children = [lhs]

    @parse_function
    def a_expr_base(self) -> AExpr:
        return AExpr(lhs=None, op=None, rhs=self.m_expr())

    @parse_function
    def a_expr(self) -> AExpr:
        lhs = self.a_expr_base()

        while True:
            try:
                with self.checkpoint():
                    _ = self.opt("whitespace")
                    op = self.expect_operator("+")
            except ParseFailedError:
                return lhs

            _ = self.opt("whitespace")

            rhs = self.m_expr()

            lhs = AExpr(lhs=lhs, op=op, rhs=rhs)
            lhs.children = self.children

            self.children = [lhs]

    @parse_function
    def shift_expr(self) -> ShiftExpr:
        return ShiftExpr(rhs=self.a_expr())

    @parse_function
    def and_expr(self) -> AndExpr:
        return AndExpr(rhs=self.shift_expr())

    @parse_function
    def xor_expr(self) -> XorExpr:
        return XorExpr(rhs=self.and_expr())

    @parse_function
    def or_expr(self) -> OrExpr:
        return OrExpr(rhs=self.xor_expr())

    @parse_function
    def comparison(self) -> Comparison:
        lhs = self.or_expr()

        ops: list[Token] | None = []
        rhs: list[OrExpr] | None = []
        while True:
            try:
                with self.checkpoint():
                    _ = self.opt("whitespace")
                    op = self.expect("operator")

                    if op.text not in {"==", "<="}:
                        raise ParseFailedError(
                            f"expected a comparison operator, found <operator {op.text}>"
                        )

                    _ = self.opt("whitespace")
                    x = self.or_expr()

                ops.append(op)
                rhs.append(x)
            except ParseFailedError:
                if len(ops) == 0:
                    ops = None
                    rhs = None

                break

        return Comparison(lhs=lhs, ops=ops, rhs=rhs)

    @parse_function
    def not_test(self) -> NotTest:
        return NotTest(x=self.comparison())

    @parse_function
    def and_test(self) -> AndTest:
        return AndTest(rhs=self.not_test())

    @parse_function
    def or_test(self) -> OrTest:
        return OrTest(rhs=self.and_test())

    @parse_function
    def starred_expression(self) -> StarredExpression:
        return StarredExpression(x=self.or_test())

    @parse_function
    def conditional_expression(self) -> ConditionalExpression:
        return ConditionalExpression(then=self.or_test())

    @parse_function
    def assignment_expression(self) -> AssignmentExpression:
        return AssignmentExpression(value=self.expression())

    @parse_function
    def expression(self) -> Expression:
        return Expression(x=self.conditional_expression())

    @parse_function
    def expression_list(self) -> ExpressionList:
        return ExpressionList(xs=[self.expression()])

    @parse_function
    def target(self) -> Target:
        return Target(x=self.expect("identifier"))

    @parse_function
    def target_list(self) -> TargetList:
        xs: list[Target] = [self.target()]

        while True:
            try:
                with self.checkpoint():
                    _ = self.opt("whitespace")
                    _ = self.expect_delimiter(",")
            except ParseFailedError:
                break

            _ = self.opt("whitespace")
            xs.append(self.target())

        # trailing comma
        try:
            with self.checkpoint():
                _ = self.opt("whitespace")
                _ = self.expect_delimiter(",")
        except ParseFailedError:
            ...

        return TargetList(xs=xs)

    @parse_function
    def assignment_stmt(self) -> AssignmentStmt:
        targets: list[TargetList] = []

        while True:
            try:
                with self.checkpoint():
                    x = self.target_list()
                    _ = self.opt("whitespace")
                    _ = self.expect_delimiter("=")

                targets.append(x)
            except ParseFailedError:
                if len(targets) == 0:
                    raise

                break

        _ = self.opt("whitespace")
        return AssignmentStmt(targets=targets, value=self.starred_expression())

    @parse_function
    def aug_target(self) -> AugTarget:
        return AugTarget(x=self.expect("identifier"))

    @parse_function
    def augmented_assignment_stmt(self) -> AugmentedAssignmentStmt:
        target = self.aug_target()

        _ = self.opt("whitespace")
        op = self.expect("delimiter")
        if op.text not in {"+="}:
            raise ParseFailedError(
                f"expected an augmented assignment operator, found <delimiter {op.text}>"
            )

        _ = self.opt("whitespace")
        return AugmentedAssignmentStmt(
            target=target, op=op, value=self.expression_list()
        )

    @parse_function
    def expression_stmt(self) -> ExpressionStmt:
        return ExpressionStmt(x=self.starred_expression())

    @parse_function
    def return_stmt(self) -> ReturnStmt:
        _ = self.expect_identifier("return")

        value: ExpressionList | None = None
        try:
            with self.checkpoint():
                _ = self.opt("whitespace")
                value = self.expression_list()
        except ParseFailedError:
            pass

        return ReturnStmt(value=value)

    @parse_function
    def simple_stmt(self) -> SimpleStmt:
        try:
            with self.checkpoint():
                return SimpleStmt(x=self.assignment_stmt())
        except ParseFailedError:
            pass

        try:
            with self.checkpoint():
                return SimpleStmt(x=self.return_stmt())
        except ParseFailedError:
            pass

        try:
            with self.checkpoint():
                return SimpleStmt(x=self.augmented_assignment_stmt())
        except ParseFailedError:
            pass

        # note(maximsmol): must be after all the statements containing keywords
        # or it will parse the keyword as an identifier
        return SimpleStmt(x=self.expression_stmt())

    @parse_function
    def stmt_list(self) -> StmtList:
        return StmtList(xs=[self.simple_stmt()])

    @parse_function
    def if_stmt(self) -> IfStmt:
        _ = self.expect_identifier("if")
        _ = self.opt("whitespace")
        cond = self.assignment_expression()
        _ = self.opt("whitespace")
        _ = self.expect_delimiter(":")
        _ = self.opt("whitespace")
        then = self.suite()

        return IfStmt(cond=cond, then=then)

    @parse_function
    def while_stmt(self) -> WhileStmt:
        _ = self.expect_identifier("while")
        _ = self.opt("whitespace")
        cond = self.assignment_expression()
        _ = self.opt("whitespace")
        _ = self.expect_delimiter(":")
        _ = self.opt("whitespace")
        loop = self.suite()

        return WhileStmt(cond=cond, loop=loop)

    @parse_function
    def defparameter(self) -> Defparameter:
        name = self.expect("identifier")

        annotation = None
        try:
            with self.checkpoint():
                _ = self.opt("whitespace")
                _ = self.expect_delimiter(":")
                _ = self.opt("whitespace")
                annotation = self.expression()
        except ParseFailedError:
            ...

        return Defparameter(name=name, annotation=annotation)

    @parse_function
    def parameter_list(self) -> ParameterList:
        regular_params: list[Defparameter] = []

        try:
            with self.checkpoint():
                regular_params.append(self.defparameter())

            while True:
                try:
                    with self.checkpoint():
                        _ = self.opt("whitespace")
                        _ = self.expect_delimiter(",")
                        _ = self.opt("whitespace")
                        regular_params.append(self.defparameter())
                except ParseFailedError:
                    break

        except ParseFailedError:
            ...

        return ParameterList(regular_params=regular_params)

    @parse_function
    def funcdef(self) -> Funcdef:
        _ = self.expect_identifier("def")
        _ = self.opt("whitespace")
        name = self.expect("identifier")
        _ = self.opt("whitespace")
        _ = self.expect_delimiter("(")
        _ = self.opt("whitespace")
        params = self.parameter_list()
        _ = self.opt("whitespace")
        _ = self.expect_delimiter(")")
        _ = self.opt("whitespace")

        return_value: Expression | None = None
        try:
            with self.checkpoint():
                _ = self.expect_delimiter("->")
                _ = self.opt("whitespace")
                return_value = self.expression()
                _ = self.opt("whitespace")
        except ParseFailedError:
            ...

        _ = self.expect_delimiter(":")
        _ = self.opt("whitespace")
        body = self.suite()

        return Funcdef(name=name, params=params, return_value=return_value, body=body)

    @parse_function
    def compound_stmt(self) -> CompoundStmt:
        try:
            with self.checkpoint():
                return CompoundStmt(x=self.if_stmt())
        except ParseFailedError:
            pass

        try:
            with self.checkpoint():
                return CompoundStmt(x=self.while_stmt())
        except ParseFailedError:
            pass

        return CompoundStmt(x=self.funcdef())

    @parse_function
    def statement(self) -> Statement:
        try:
            with self.checkpoint():
                x = self.stmt_list()
                _ = self.expect("newline")
                return Statement(x=x)
        except ParseFailedError:
            pass

        return Statement(x=self.compound_stmt())

    @parse_function
    def suite(self) -> Suite:
        try:
            with self.checkpoint():
                x = self.stmt_list()
                _ = self.expect("newline")
                return Suite(xs=x)
        except ParseFailedError:
            pass

        _ = self.expect("newline")
        _ = self.expect("indent")

        xs: list[Statement] = []
        while True:
            try:
                with self.checkpoint():
                    _ = self.opt("whitespace")
                    xs.append(self.statement())
            except ParseFailedError:
                break

        if len(xs) == 0:
            raise ParseFailedError("expected at least one statement")

        while True:
            nl = self.opt("nl")
            if nl is None:
                break
        _ = self.expect("dedent")

        return Suite(xs=xs)

    @parse_function
    def file_input(self) -> FileInput:
        xs: list[Statement] = []
        while True:
            end = self.opt("endmarker")
            if end is not None:
                break
            xs.append(self.statement())

        return FileInput(xs=xs)

    @contextmanager
    def parse_wrapper(self) -> Generator[None]:
        try:
            yield
        except Exception as e:
            idx = None

            l, cursor = self.lex.debug_pos(idx=idx)
            e.add_note(l)
            e.add_note(cursor)
            raise

    def parse(self) -> FileInput:
        with self.parse_wrapper():
            return self.file_input()

    def parse_expr(self) -> StarredExpression:
        with self.parse_wrapper():
            return self.starred_expression()
