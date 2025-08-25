import re
import token
from copy import copy
from dataclasses import dataclass
from tokenize import TokenInfo
from typing import override

from .grammar import integers as g_int
from .grammar import operators as g_op


@dataclass(kw_only=True)
class Pos:
    idx: int
    line: int
    col: int

    @override
    def __str__(self) -> str:
        return f"{self.line}:{self.col}#{self.idx}"

    def tuple(self) -> tuple[int, int]:
        return (self.line, self.col - 1)


@dataclass(kw_only=True)
class Token:
    type: str
    start: Pos
    end: Pos
    text: str

    @override
    def __str__(self) -> str:
        return f"{self.type}@{self.start}{self.text!r}"

    def token_info(self) -> TokenInfo | None:
        text = self.text
        end = self.end
        if self.type == "newline":
            end = copy(end)
            end.line -= 1
            end.col = self.start.col + len(self.text)

            text = ""

        return TokenInfo(
            type={
                "newline": token.NEWLINE,
                "decinteger": token.NUMBER,
                "operator": token.OP,
                "endmarker": token.ENDMARKER,
            }[self.type],
            string=text,
            start=self.start.tuple(),
            end=end.tuple(),
            line="",
        )


toks: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[ \t\f]+"), "whitespace"),
    (re.compile(r"\n|\r\n|\r"), "newline"),
    (g_int.decinteger_re, "decinteger"),
    (g_op.operator_re, "operator"),
]


class Lexer:
    def __init__(self, *, data: str) -> None:
        self.data: str = data
        self.pos: Pos = Pos(idx=0, line=1, col=1)

        if self.data[-1] not in "\n\r":
            self.data += "\n"

    def debug_pos(self, *, idx: int | None = None) -> tuple[str, str]:
        if idx is None:
            idx = self.pos.idx

        # todo(maximsmol): support \r
        line_start = max(self.data.rfind("\n", 0, idx), 0)
        line_end = min(self.data.find("\n", idx), len(self.data))

        # no more than 80 characters total, keep at least 10 characters after the cursor, if there are enough
        suffix_len = min(line_end - idx, 10)

        prefix = ""
        suffix = ""

        min_start = idx - (80 - suffix_len)
        if line_start < min_start:
            line_start = min_start + 1
            prefix = "…"

        if line_end > line_start + 80:
            line_end = line_start + 79
            suffix = "…"

        return prefix + self.data[line_start:line_end] + suffix, " " * (
            idx - line_start - 1 + len(prefix)
        ) + "^"

    def next(self) -> Token:
        if self.pos.idx == len(self.data):
            return Token(type="endmarker", start=self.pos, end=self.pos, text="")

        for r, t in toks:
            m = r.match(self.data, self.pos.idx)
            if m is None:
                continue

            start = copy(self.pos)
            match_str = m.group(0)
            self.pos.idx += len(match_str)

            # todo(maximsmol): support \n, \n\r, and \r
            line_count = match_str.count("\n")
            self.pos.line += line_count
            if line_count > 0:
                self.pos.col = len(match_str) - match_str.rindex("\n")
            else:
                self.pos.col += len(match_str)

            return Token(type=t, start=start, end=copy(self.pos), text=match_str)

        raise RuntimeError(f"unknown token at {self.pos}")
