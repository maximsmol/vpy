from vpy.lex import Token

from .parse import (
    AExpr,
    AndExpr,
    AssignmentStmt,
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


class Interpreter:
    def __init__(self) -> None:
        self.locals: dict[str, object] = {}

    def eval(self, x: Node) -> object:
        if isinstance(x, StarredExpression):
            return self.eval(x.x)

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

            assert x.op.text == "+"

            return self.eval(x.lhs) + self.eval(x.rhs)

        if isinstance(x, MExpr):
            return self.eval(x.rhs)

        if isinstance(x, UExpr):
            return self.eval(x.x)

        if isinstance(x, Power):
            return self.eval(x.x)

        if isinstance(x, Primary):
            return self.eval(x.x)

        if isinstance(x, Atom):
            if isinstance(x.x, Token):
                assert x.x.type == "identifier"
                return self.locals[x.x.nfkd()]

            return self.eval(x.x)

        if isinstance(x, AstLiteral):
            assert x.x.type == "decinteger"
            return int(x.x.text)

        raise NotImplementedError(f"unknown node: {x.type}")

    def exec(self, x: Node) -> None:
        if isinstance(x, FileInput):
            for s in x.xs:
                self.exec(s)
            return

        if isinstance(x, Statement):
            self.exec(x.x)
            return

        if isinstance(x, StmtList):
            for s in x.xs:
                self.exec(s)
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

        raise NotImplementedError(f"unknown node: {x.type}")
