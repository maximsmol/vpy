import ast
from collections.abc import Callable, Generator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import wraps
from typing import Concatenate, Generic, Literal, TypeVar, override

from vpy.lex import Lexer, Token

T_co = TypeVar("T_co", default=str, bound=str, covariant=True)


@dataclass(kw_only=True)
class Node(Generic[T_co]):
    type: T_co
    children: "list[Node[str] | Token]" = field(default_factory=list)

    def to_str(self, *, indent: str = "", visited: set[int] | None = None) -> str:
        if visited is None:
            visited = set()

        if id(self) in visited:
            return f"<recursion: {self.type}#{id(self)}>"
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

    def unparse(self, *, parens: bool = False) -> str:
        res: list[str] = []

        for x in self.children:
            if isinstance(x, Token):
                res.append(x.text)
                continue

            res.append(x.unparse(parens=parens))

        return "".join(res)

    def to_ast(self) -> ast.AST:
        raise NotImplementedError(f"{self.type!r} has no `to_ast` implementation")


@dataclass(kw_only=True)
class AstLiteral(Node[Literal["literal"]]):
    x: Token

    @override
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
class Atom(Node[Literal["atom"]]):
    x: AstLiteral | Token

    @override
    def to_ast(self) -> ast.Constant | ast.Name:
        if isinstance(self.x, Token):
            assert self.x.type == "identifier"
            return ast.Name(self.x.nfkd())

        return self.x.to_ast()


@dataclass(kw_only=True)
class Primary(Node[Literal["primary"]]):
    x: Atom

    @override
    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class Power(Node[Literal["power"]]):
    x: Primary

    @override
    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class UExpr(Node[Literal["u_expr"]]):
    x: Power

    @override
    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class MExpr(Node[Literal["m_expr"]]):
    lhs: "MExpr | None"
    op: Token | None
    rhs: UExpr

    @override
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
class AExpr(Node[Literal["a_expr"]]):
    lhs: "AExpr | None"
    op: Token | None
    rhs: MExpr

    @override
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
class ShiftExpr(Node[Literal["shift_expr"]]):
    rhs: AExpr

    @override
    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class AndExpr(Node[Literal["and_expr"]]):
    rhs: ShiftExpr

    @override
    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class XorExpr(Node[Literal["xor_expr"]]):
    rhs: AndExpr

    @override
    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class OrExpr(Node[Literal["or_expr"]]):
    rhs: XorExpr

    @override
    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class Comparison(Node[Literal["comparison"]]):
    lhs: OrExpr
    ops: list[Token] | None
    rhs: list[OrExpr] | None

    @override
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
class NotTest(Node[Literal["not_test"]]):
    x: Comparison

    @override
    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class AndTest(Node[Literal["and_test"]]):
    rhs: NotTest

    @override
    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class OrTest(Node[Literal["or_test"]]):
    rhs: AndTest

    @override
    def to_ast(self) -> ast.expr:
        return self.rhs.to_ast()


@dataclass(kw_only=True)
class StarredExpression(Node[Literal["starred_expression"]]):
    x: OrTest

    @override
    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class ExpressionStmt(Node[Literal["expression_stmt"]]):
    x: StarredExpression

    @override
    def to_ast(self) -> ast.Expr:
        return ast.Expr(value=self.x.to_ast())


@dataclass(kw_only=True)
class Target(Node[Literal["target"]]):
    x: Token

    @override
    def to_ast(self) -> ast.Name:
        return ast.Name(id=self.x.nfkd(), ctx=ast.Store())


@dataclass(kw_only=True)
class TargetList(Node[Literal["target_list"]]):
    xs: list[Target]

    @override
    def to_ast(self) -> ast.Name:
        assert len(self.xs) == 1
        return self.xs[0].to_ast()


@dataclass(kw_only=True)
class AssignmentStmt(Node[Literal["assignment_stmt"]]):
    targets: list[TargetList]
    value: StarredExpression

    @override
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
class AugTarget(Node[Literal["augtarget"]]):
    x: Token

    @override
    def to_ast(self) -> ast.Name:
        return ast.Name(id=self.x.nfkd(), ctx=ast.Store())


@dataclass(kw_only=True)
class ConditionalExpression(Node[Literal["conditional_expression"]]):
    then: OrTest

    @override
    def to_ast(self) -> ast.expr:
        return self.then.to_ast()


@dataclass(kw_only=True)
class AssignmentExpression(Node[Literal["assignment_expression"]]):
    value: "Expression"

    @override
    def to_ast(self) -> ast.expr:
        return self.value.to_ast()


@dataclass(kw_only=True)
class Expression(Node[Literal["expression"]]):
    x: ConditionalExpression

    @override
    def to_ast(self) -> ast.expr:
        return self.x.to_ast()


@dataclass(kw_only=True)
class ExpressionList(Node[Literal["expression_list"]]):
    xs: list[Expression]

    @override
    def to_ast(self) -> ast.AST:
        return self.xs[0].to_ast()


@dataclass(kw_only=True)
class AugmentedAssignmentStmt(Node[Literal["augmented_assignment_stmt"]]):
    target: AugTarget
    op: Token
    value: ExpressionList

    @override
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
class SimpleStmt(Node[Literal["simple_stmt"]]):
    x: ExpressionStmt | AssignmentStmt | AugmentedAssignmentStmt

    @override
    def to_ast(self) -> ast.stmt:
        return self.x.to_ast()


@dataclass(kw_only=True)
class StmtList(Node[Literal["stmt_list"]]):
    xs: list[SimpleStmt]

    @override
    def to_ast(self) -> ast.stmt:
        assert len(self.xs) == 1
        return self.xs[0].to_ast()


@dataclass(kw_only=True)
class IfStmt(Node[Literal["if_stmt"]]):
    cond: AssignmentExpression
    then: "Suite"

    @override
    def to_ast(self) -> ast.If:
        return ast.If(test=self.cond.to_ast(), body=[self.then.to_ast()])


@dataclass(kw_only=True)
class WhileStmt(Node[Literal["while_stmt"]]):
    cond: AssignmentExpression
    loop: "Suite"

    @override
    def to_ast(self) -> ast.While:
        return ast.While(test=self.cond.to_ast(), body=[self.loop.to_ast()])


@dataclass(kw_only=True)
class CompoundStmt(Node[Literal["compound_stmt"]]):
    x: IfStmt | WhileStmt

    @override
    def to_ast(self) -> ast.stmt:
        return self.x.to_ast()


@dataclass(kw_only=True)
class Statement(Node[Literal["statement"]]):
    x: StmtList | CompoundStmt

    @override
    def to_ast(self) -> ast.stmt:
        return self.x.to_ast()


@dataclass(kw_only=True)
class Suite(Node[Literal["suite"]]):
    xs: StmtList | list[Statement]

    @override
    def to_ast(self) -> ast.stmt:
        if isinstance(self.xs, StmtList):
            return self.xs.to_ast()

        assert len(self.xs) == 1
        return self.xs[0].to_ast()


@dataclass(kw_only=True)
class FileInput(Node[Literal["file_input"]]):
    xs: list[Statement]

    @override
    def to_ast(self) -> ast.Module:
        body: list[ast.stmt] = []

        for x in self.xs:
            cur = x.to_ast()
            assert isinstance(cur, ast.stmt)
            body.append(cur)

        return ast.Module(body=body)


type ParseFunctionIn[**T, R] = Callable[Concatenate[Parser, T], R]
type ParseFunctionOut[**T, R] = Callable[Concatenate[Parser, T], R]


def parse_function[**T, R: Node[str]](
    f: ParseFunctionIn[T, R],
) -> ParseFunctionOut[T, R]:
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
        self.token_stack: list[Token] = []
        self.children: list[Node[str] | Token] = []

    @contextmanager
    def checkpoint(self) -> Generator[None]:
        old = deepcopy(self)

        try:
            yield
        except Exception:
            self.lex = old.lex
            self.token_stack = old.token_stack
            self.children = old.children

            raise

    def tok(self) -> Token:
        if len(self.token_stack) > 0:
            res = self.token_stack.pop()
        else:
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
            raise ParseFailedError(f"expected <{typ}>, found <{res.type}>")

        return res

    def expect_operator(self, text: str) -> Token:
        res = self.tok()
        if res.type != "operator":
            raise ParseFailedError(f"expected <operator {text}>, found <{res.type}>")

        if res.text != text:
            raise ParseFailedError(
                f"expected <operator {text}>, found <operator {res.text}>"
            )

        return res

    def expect_delimiter(self, text: str) -> Token:
        res = self.tok()
        if res.type != "delimiter":
            raise ParseFailedError(f"expected <delimiter {text}>, found <{res.type}>")

        if res.text != text:
            raise ParseFailedError(
                f"expected <delimiter {text}>, found <delimiter {res.text}>"
            )

        return res

    def expect_identifier(self, text: str) -> Token:
        res = self.tok()
        if res.type != "identifier":
            raise ParseFailedError(f"expected <identifier {text}>, found <{res.type}>")

        if res.text != text:
            raise ParseFailedError(
                f"expected <identifier {text}>, found <identifier {res.text}>"
            )

        return res

    @parse_function
    def literal(self) -> AstLiteral:
        return AstLiteral(type="literal", x=self.expect("decinteger"))

    @parse_function
    def atom(self) -> Atom:
        ident = self.opt("identifier")
        if ident is not None:
            if ident.text in {"True", "False", "None"}:
                return Atom(type="atom", x=AstLiteral(type="literal", x=ident))

            return Atom(type="atom", x=ident)

        return Atom(type="atom", x=self.literal())

    @parse_function
    def primary(self) -> Primary:
        return Primary(type="primary", x=self.atom())

    @parse_function
    def power(self) -> Power:
        return Power(type="power", x=self.primary())

    @parse_function
    def u_expr(self) -> UExpr:
        return UExpr(type="u_expr", x=self.power())

    @parse_function
    def m_expr_base(self) -> MExpr:
        return MExpr(type="m_expr", lhs=None, op=None, rhs=self.u_expr())

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

            lhs = MExpr(type="m_expr", lhs=lhs, op=op, rhs=rhs)
            lhs.children = self.children

            self.children = [lhs]

    @parse_function
    def a_expr_base(self) -> AExpr:
        return AExpr(type="a_expr", lhs=None, op=None, rhs=self.m_expr())

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

            lhs = AExpr(type="a_expr", lhs=lhs, op=op, rhs=rhs)
            lhs.children = self.children

            self.children = [lhs]

    @parse_function
    def shift_expr(self) -> ShiftExpr:
        return ShiftExpr(type="shift_expr", rhs=self.a_expr())

    @parse_function
    def and_expr(self) -> AndExpr:
        return AndExpr(type="and_expr", rhs=self.shift_expr())

    @parse_function
    def xor_expr(self) -> XorExpr:
        return XorExpr(type="xor_expr", rhs=self.and_expr())

    @parse_function
    def or_expr(self) -> OrExpr:
        return OrExpr(type="or_expr", rhs=self.xor_expr())

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

        return Comparison(type="comparison", lhs=lhs, ops=ops, rhs=rhs)

    @parse_function
    def not_test(self) -> NotTest:
        return NotTest(type="not_test", x=self.comparison())

    @parse_function
    def and_test(self) -> AndTest:
        return AndTest(type="and_test", rhs=self.not_test())

    @parse_function
    def or_test(self) -> OrTest:
        return OrTest(type="or_test", rhs=self.and_test())

    @parse_function
    def starred_expression(self) -> StarredExpression:
        return StarredExpression(type="starred_expression", x=self.or_test())

    @parse_function
    def conditional_expression(self) -> ConditionalExpression:
        return ConditionalExpression(type="conditional_expression", then=self.or_test())

    @parse_function
    def assignment_expression(self) -> AssignmentExpression:
        return AssignmentExpression(
            type="assignment_expression", value=self.expression()
        )

    @parse_function
    def expression(self) -> Expression:
        return Expression(type="expression", x=self.conditional_expression())

    @parse_function
    def expression_list(self) -> ExpressionList:
        return ExpressionList(type="expression_list", xs=[self.expression()])

    @parse_function
    def target(self) -> Target:
        return Target(type="target", x=self.expect("identifier"))

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

        return TargetList(type="target_list", xs=xs)

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
        return AssignmentStmt(
            type="assignment_stmt", targets=targets, value=self.starred_expression()
        )

    @parse_function
    def aug_target(self) -> AugTarget:
        return AugTarget(type="augtarget", x=self.expect("identifier"))

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
            type="augmented_assignment_stmt",
            target=target,
            op=op,
            value=self.expression_list(),
        )

    @parse_function
    def expression_stmt(self) -> ExpressionStmt:
        return ExpressionStmt(type="expression_stmt", x=self.starred_expression())

    @parse_function
    def simple_stmt(self) -> SimpleStmt:
        try:
            with self.checkpoint():
                return SimpleStmt(type="simple_stmt", x=self.assignment_stmt())
        except ParseFailedError:
            pass

        try:
            with self.checkpoint():
                return SimpleStmt(
                    type="simple_stmt", x=self.augmented_assignment_stmt()
                )
        except ParseFailedError:
            pass

        return SimpleStmt(type="simple_stmt", x=self.expression_stmt())

    @parse_function
    def stmt_list(self) -> StmtList:
        return StmtList(type="stmt_list", xs=[self.simple_stmt()])

    @parse_function
    def if_stmt(self) -> IfStmt:
        _ = self.expect_identifier("if")
        _ = self.opt("whitespace")
        cond = self.assignment_expression()
        _ = self.opt("whitespace")
        _ = self.expect_delimiter(":")
        _ = self.opt("whitespace")
        then = self.suite()

        return IfStmt(type="if_stmt", cond=cond, then=then)

    @parse_function
    def while_stmt(self) -> WhileStmt:
        _ = self.expect_identifier("while")
        _ = self.opt("whitespace")
        cond = self.assignment_expression()
        _ = self.opt("whitespace")
        _ = self.expect_delimiter(":")
        _ = self.opt("whitespace")
        loop = self.suite()

        return WhileStmt(type="while_stmt", cond=cond, loop=loop)

    @parse_function
    def compound_stmt(self) -> CompoundStmt:
        try:
            with self.checkpoint():
                return CompoundStmt(type="compound_stmt", x=self.if_stmt())
        except ParseFailedError:
            pass

        return CompoundStmt(type="compound_stmt", x=self.while_stmt())

    @parse_function
    def statement(self) -> Statement:
        try:
            with self.checkpoint():
                x = self.stmt_list()
                _ = self.expect("newline")
                return Statement(type="statement", x=x)
        except ParseFailedError:
            pass

        return Statement(type="statement", x=self.compound_stmt())

    @parse_function
    def suite(self) -> Suite:
        try:
            with self.checkpoint():
                x = self.stmt_list()
                _ = self.expect("newline")
                return Suite(type="suite", xs=x)
        except ParseFailedError:
            pass

        _ = self.expect("newline")
        _ = self.expect("indent")

        xs: list[Statement] = []
        while True:
            try:
                with self.checkpoint():
                    xs.append(self.statement())
            except ParseFailedError:
                break

        if len(xs) == 0:
            raise ParseFailedError("expected at least one statement")

        _ = self.expect("dedent")

        return Suite(type="suite", xs=xs)

    @parse_function
    def file_input(self) -> FileInput:
        xs: list[Statement] = []
        while True:
            end = self.opt("endmarker")
            if end is not None:
                break
            xs.append(self.statement())

        return FileInput(type="file_input", xs=xs)

    @contextmanager
    def parse_wrapper(self) -> Generator[None]:
        try:
            yield
        except Exception as e:
            idx = None
            if len(self.token_stack) > 0:
                idx = self.token_stack[-1].start.idx

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
