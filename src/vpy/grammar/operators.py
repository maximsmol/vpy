import re

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

operator_re = re.compile(operator, re.VERBOSE)
