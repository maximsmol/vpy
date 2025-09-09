import regex

delimiters = r"""
(
  \(
| \)
| \[
| \]
| \{
| \}
| ,
| :
| !
| \.
| ;
| @
| =
| ->
| \+=
| -=
| \*=
| /=
| //=
| %=
| @=
| &=
| \|=
| \^=
| >>=
| <<=
| \*\*=
)
"""

delimiters_re = regex.compile(delimiters, regex.VERBOSE)
