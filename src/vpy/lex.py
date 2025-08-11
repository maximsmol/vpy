keywords = [
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
]


class Lexer:
    def __init__(self, *, data: str) -> None:
        self.data: str = data
        self.pos: int = 0
        self.col: int = 1
        self.line: int = 1

    def next(self) -> str:
        return "<unk>"
