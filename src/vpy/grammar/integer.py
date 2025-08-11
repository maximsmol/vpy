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
decinteger_re = re.compile(decinteger)

bininteger = rf"""
(
  0
  [bB]
  (_? {bindigit})+
)
"""
bininteger_re = re.compile(bininteger)

octinteger = rf"""
(
  0
  [oO]
  (_? {octdigit})+
)
"""
octinteger_re = re.compile(octinteger)

hexinteger = rf"""
(
  0
  [xX]
  (_? {hexdigit})+
)
"""
hexinteger_re = re.compile(hexinteger)
