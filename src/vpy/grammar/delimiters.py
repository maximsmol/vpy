import regex

delimiters_short = r"[()[\]{},:!.;@=]"
delimiters_short_re = regex.compile(delimiters_short, regex.VERBOSE)

delimiters_long = r"""
(
  ->
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

delimiters_long_re = regex.compile(delimiters_long, regex.VERBOSE)
