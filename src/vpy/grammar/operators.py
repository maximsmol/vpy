import regex

operator = r"""
(
  \+
| <<
| <
| -
| >>
| >
| \*
| &
| <=
| \*\*
| \|
| >=
| /
| \^
| ==
| //
| ~
| !=
| %
| :=
| @
)
"""

operator_re = regex.compile(operator, regex.VERBOSE)
