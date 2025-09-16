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


class Interpreter:
    def __init__(self) -> None:
        self.locals: dict[str, object] = {}

    def eval(self, x: Node) -> object:
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
                        cond = cur == rhs
                    case "<=":
                        assert isinstance(cur, int)
                        assert isinstance(rhs, int)
                        cond = cur <= rhs
                    case _:
                        raise RuntimeError(
                            f"unsupported comparison operator: {op.text}"
                        )

                if not cond:
                    return False
                cur = rhs

            return True

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
            assert isinstance(lhs, int)

            rhs = self.eval(x.rhs)
            assert isinstance(rhs, int)

            match x.op.text:
                case "+":
                    return lhs + rhs

                case _:
                    raise NotImplementedError(f"unsupported operator: {x.op.text}")

        if isinstance(x, MExpr):
            if x.lhs is None:
                return self.eval(x.rhs)

            assert x.op is not None

            lhs = self.eval(x.lhs)
            assert isinstance(lhs, int)

            rhs = self.eval(x.rhs)
            assert isinstance(rhs, int)

            match x.op.text:
                case "%":
                    return lhs % rhs

                case "*":
                    return lhs * rhs

                case _:
                    raise NotImplementedError(f"unsupported operator: {x.op.text}")

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
            match x.x.type:
                case "decinteger":
                    return int(x.x.text)

                case "identifier":
                    match x.x.text:
                        case "True":
                            return True
                        case "False":
                            return False
                        case "None":
                            return None
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
            if cond is True:
                self.exec(x.then)
            return

        if isinstance(x, WhileStmt):
            cond = self.eval(x.cond)
            while cond:
                self.exec(x.loop)
                cond = self.eval(x.cond)
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
            assert isinstance(cur, int)

            val = self.eval(x.value)
            assert isinstance(val, int)

            match x.op.text:
                case "+=":
                    self.locals[name] = cur + val
                case _:
                    raise NotImplementedError(
                        f"unsupported augmented assignment operator: {x.op.text}"
                    )

            return

        raise NotImplementedError(f"unknown node: {x.type}")
