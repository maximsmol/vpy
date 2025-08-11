import re

nonzerodigit = r"([1-9])"
digit = r"([0-9])"
bindigit = r"([01])"
octdigit = r"([0-7])"
hexdigit = rf"({digit} | [a-fA-F])"

# integer = decinteger | bininteger | octinteger | hexinteger
decinteger = rf"""
(
  {nonzerodigit} (_? {digit})*
  |
  0+ (_? 0)*
)
"""
decinteger_re = re.compile(decinteger, re.VERBOSE)

bininteger = rf"""
(
  0
  [bB]
  (_? {bindigit})+
)
"""
bininteger_re = re.compile(bininteger, re.VERBOSE)

octinteger = rf"""
(
  0
  [oO]
  (_? {octdigit})+
)
"""
octinteger_re = re.compile(octinteger, re.VERBOSE)

hexinteger = rf"""
(
  0
  [xX]
  (_? {hexdigit})+
)
"""
hexinteger_re = re.compile(hexinteger, re.VERBOSE)
