import token
from copy import copy
from dataclasses import dataclass
from tokenize import TokenInfo
from typing import override
import unicodedata

import regex

from .grammar import delimiters as g_delim
from .grammar import indentifiers as g_id
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

    def nfkd(self) -> str:
        return unicodedata.normalize("NFKD", self.text)

    def token_info(self) -> TokenInfo | None:
        text = self.text
        end = self.end
        if self.type == "newline":
            end = copy(end)
            end.line -= 1
            end.col = self.start.col + max(len(self.text), 1)

        return TokenInfo(
            type={
                "newline": token.NEWLINE,
                "decinteger": token.NUMBER,
                "operator": token.OP,
                "endmarker": token.ENDMARKER,
                "identifier": token.NAME,
                "delimiter": token.OP,
            }[self.type],
            string=text,
            start=self.start.tuple(),
            end=end.tuple(),
            line="",
        )


toks: list[tuple[regex.Pattern[str], str]] = [
    (regex.compile(r"[ \t\f]+"), "whitespace"),
    (regex.compile(r"\n|\r\n|\r"), "newline"),
    (g_int.decinteger_re, "decinteger"),
    (g_op.operator_re, "operator"),
    (g_id.identifier_re, "identifier"),
    (g_delim.delimiters_re, "delimiter"),
]


class Lexer:
    def __init__(self, *, data: str) -> None:
        self.data: str = data
        self.pos: Pos = Pos(idx=0, line=1, col=1)
        self.last_was_newline = False

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
            if not self.last_was_newline:
                self.last_was_newline = True
                start = copy(self.pos)

                self.pos.line += 1
                self.pos.col = 1

                return Token(type="newline", start=start, end=copy(self.pos), text="")

            return Token(
                type="endmarker", start=copy(self.pos), end=copy(self.pos), text=""
            )

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

            self.last_was_newline = t == "newline"
            return Token(type=t, start=start, end=copy(self.pos), text=match_str)

        e = RuntimeError(f"unknown token at {self.pos}")
        l, cursor = self.debug_pos(idx=self.pos.idx + 1)
        e.add_note(l)
        e.add_note(cursor)
        raise e
