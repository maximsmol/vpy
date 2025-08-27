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


class Interpreter:
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

        raise NotImplementedError(f"unknown node: {x.type}")
