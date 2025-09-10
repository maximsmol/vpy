import regex

operators_short = r"[+<\->*&|/^~%@]"
operator_short_re = regex.compile(operators_short, regex.VERBOSE)

operators_long = r"""
(
  <<
| >>
| <=
| \*\*
| >=
| ==
| //
| !=
| :=
)
"""


operator_long_re = regex.compile(operators_long, regex.VERBOSE)
