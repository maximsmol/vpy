import token
import unicodedata
from copy import copy
from dataclasses import dataclass
from tokenize import TokenInfo
from typing import override

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
        if self.type in {"newline", "nl"}:
            end = copy(end)
            end.line -= 1
            end.col = self.start.col + max(len(self.text), 1)

        return TokenInfo(
            type={
                "nl": token.NL,
                "newline": token.NEWLINE,
                "decinteger": token.NUMBER,
                "operator": token.OP,
                "endmarker": token.ENDMARKER,
                "identifier": token.NAME,
                "delimiter": token.OP,
                "indent": token.INDENT,
                "dedent": token.DEDENT,
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
    (g_delim.delimiters_long_re, "delimiter"),
    (g_op.operator_long_re, "operator"),
    (g_delim.delimiters_short_re, "delimiter"),
    (g_op.operator_short_re, "operator"),
    (g_id.identifier_re, "identifier"),
]


class Lexer:
    def __init__(self, *, data: str) -> None:
        self.data: str = data
        self.pos: Pos = Pos(idx=0, line=1, col=1)
        self.last_was_newline: bool = False

        self.token_stack: list[Token] = []
        self.indentation_stack: list[int] = [0]

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
        if len(self.token_stack) > 0:
            return self.token_stack.pop()

        return self.lex_more()

    def lex_more(self) -> Token:
        if self.pos.idx == len(self.data):
            if not self.last_was_newline:
                start = copy(self.pos)

                self.pos.line += 1
                self.pos.col = 1

                self.last_was_newline = True

                return Token(type="newline", start=start, end=copy(self.pos), text="")

            endmarker = Token(
                type="endmarker", start=copy(self.pos), end=copy(self.pos), text=""
            )
            self.token_stack.append(endmarker)

            while len(self.indentation_stack) > 1:
                _ = self.indentation_stack.pop()
                self.token_stack.append(
                    Token(
                        type="dedent", start=copy(self.pos), end=copy(self.pos), text=""
                    )
                )

            return self.token_stack.pop()

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

            end = copy(self.pos)

            start_of_line = start.col == 1
            self.last_was_newline = t == "newline"

            cur = Token(type=t, start=start, end=end, text=match_str)

            if start_of_line:
                if cur.type == "newline":
                    cur.type = "nl"
                else:
                    cur_level = self.indentation_stack[-1]
                    level = len(match_str) if cur.type == "whitespace" else 0

                    if level == cur_level:
                        pass
                    elif level > cur_level:
                        self.indentation_stack.append(level)
                        cur.type = "indent"
                    else:
                        assert level in self.indentation_stack

                        pos = start
                        if cur.type == "whitespace":
                            pos = end

                        dedent = Token(
                            type="dedent", start=copy(pos), end=copy(pos), text=""
                        )

                        self.token_stack.append(cur)

                        while self.indentation_stack[-1] != level:
                            self.token_stack.append(dedent)
                            _ = self.indentation_stack.pop()

                        return self.token_stack.pop()

            return cur

        e = RuntimeError(f"unknown token at {self.pos}")
        l, cursor = self.debug_pos(idx=self.pos.idx + 1)
        e.add_note(l)
        e.add_note(cursor)
        raise e
