from collections.abc import Callable
from dataclasses import dataclass, field, fields
from functools import wraps
from typing import Concatenate, Generic, Literal, TypeVar, override

from vpy.lex import Lexer, Token

T_co = TypeVar("T_co", default=str, bound=str, covariant=True)


@dataclass(kw_only=True)
class Node(Generic[T_co]):
    type: T_co
    children: "list[Node[str] | Token]" = field(default_factory=list)

    def to_str(self, *, indent: str = "") -> str:
        node_fields: dict[int, str] = {}

        fs = fields(self)
        for f in fs:
            if f.name in {"type", "children"}:
                continue

            val = getattr(self, f.name)
            if not isinstance(val, Node):
                continue

            node_fields[id(val)] = f.name

        children_lines: list[str] = []
        for x in self.children:
            field_name = node_fields.get(id(x))
            if field_name is None:
                if isinstance(x, Node):
                    children_lines.append(f"{indent}  {x.to_str(indent=f'{indent}  ')}")
                else:
                    children_lines.append(f"{indent}  {x}")
            else:
                assert isinstance(x, Node)
                children_lines.extend([
                    f"{indent}  #{field_name}: " + x.to_str(indent=f"{indent}  ")
                ])

        lines: list[str] = [f"{self.type}{{", *children_lines]
        lines.append(f"{indent}}}")

        return "\n".join(lines)

    @override
    def __str__(self) -> str:
        return self.to_str()


@dataclass(kw_only=True)
class Power(Node[Literal["power"]]):
    x: Node[Literal["atom"]]


@dataclass(kw_only=True)
class AExpr(Node[Literal["a_expr"]]):
    lhs: Node[Literal["m_expr"]]
    rhs: Node[Literal["m_expr"]] | None


def parse_function[**T, R: Node[str]](
    f: Callable[Concatenate["Parser", T], R],
) -> Callable[Concatenate["Parser", T], R]:
    @wraps(f)
    def res(self: "Parser", *args: T.args, **kwargs: T.kwargs) -> R:
        siblings = self.children
        try:
            self.children = []
            res = f(self, *args, **kwargs)
            res.children = self.children

            siblings.append(res)

            return res
        finally:
            self.children = siblings

    return res


class Parser:
    def __init__(self, *, lex: Lexer) -> None:
        self.lex: Lexer = lex
        self.token_stack: list[Token] = []
        self.children: list[Node[str] | Token] = []

    def tok(self) -> Token:
        if len(self.token_stack) > 0:
            res = self.token_stack.pop()
        else:
            res = self.lex.next()

        self.children.append(res)
        return res

    def untok(self) -> None:
        x = self.children[-1]
        assert isinstance(x, Token)

        self.token_stack.append(x)
        _ = self.children.pop()

    def opt(self, typ: str) -> Token | None:
        x = self.tok()
        if x.type != typ:
            self.untok()
            return None

        return x

    def expect(self, typ: str) -> Token:
        res = self.opt(typ)
        if res is None:
            raise RuntimeError(f"expected <{typ}>")

        return res

    @parse_function
    def literal(self) -> Node[Literal["literal"]]:
        _ = self.expect("decinteger")
        return Node(type="literal")

    @parse_function
    def atom(self) -> Node[Literal["atom"]]:
        _ = self.literal()
        return Node(type="atom")

    @parse_function
    def primary(self) -> Node[Literal["primary"]]:
        _ = self.atom()
        return Node(type="primary")

    @parse_function
    def power(self) -> Power:
        return Power(type="power", x=self.atom())

    @parse_function
    def u_expr(self) -> Node[Literal["u_expr"]]:
        _ = self.power()
        return Node(type="u_expr")

    @parse_function
    def m_expr(self) -> Node[Literal["m_expr"]]:
        return Node(type="m_expr", children=[self.u_expr()])

    @parse_function
    def a_expr(self) -> AExpr:
        lhs = self.m_expr()
        rhs: Node[Literal["m_expr"]] | None = None

        _ = self.opt("whitespace")

        op = self.opt("operator")
        if op is not None:
            if op.text == "+":
                _ = self.opt("whitespace")
                rhs = self.m_expr()
            else:
                self.untok()

        return AExpr(type="a_expr", lhs=lhs, rhs=rhs)

    @parse_function
    def shift_expr(self) -> Node[Literal["shift_expr"]]:
        _ = self.a_expr()
        return Node(type="shift_expr")

    @parse_function
    def and_expr(self) -> Node[Literal["and_expr"]]:
        _ = self.shift_expr()
        return Node(type="and_expr")

    @parse_function
    def xor_expr(self) -> Node[Literal["xor_expr"]]:
        _ = self.and_expr()
        return Node(type="xor_expr")

    @parse_function
    def or_expr(self) -> Node[Literal["or_expr"]]:
        _ = self.xor_expr()
        return Node(type="or_expr")

    @parse_function
    def starred_expression(self) -> Node[Literal["starred_expression"]]:
        _ = self.or_expr()
        return Node(type="starred_expression")

    @parse_function
    def expression_stmt(self) -> Node[Literal["expression_stmt"]]:
        _ = self.starred_expression()
        return Node(type="expression_stmt")

    @parse_function
    def simple_stmt(self) -> Node[Literal["simple_stmt"]]:
        _ = self.expression_stmt()
        return Node(type="simple_stmt")

    @parse_function
    def stmt_list(self) -> Node[Literal["stmt_list"]]:
        _ = self.simple_stmt()
        return Node(type="stmt_list")

    @parse_function
    def statement(self) -> Node[Literal["statement"]]:
        _ = self.stmt_list()
        return Node(type="statement")

    @parse_function
    def file_input(self) -> Node[Literal["file_input"]]:
        _ = self.statement()
        return Node(type="file_input")
