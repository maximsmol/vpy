import regex

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
decinteger_re = regex.compile(decinteger, regex.VERBOSE)

bininteger = rf"""
(
  0
  [bB]
  (_? {bindigit})+
)
"""
bininteger_re = regex.compile(bininteger, regex.VERBOSE)

octinteger = rf"""
(
  0
  [oO]
  (_? {octdigit})+
)
"""
octinteger_re = regex.compile(octinteger, regex.VERBOSE)

hexinteger = rf"""
(
  0
  [xX]
  (_? {hexdigit})+
)
"""
hexinteger_re = regex.compile(hexinteger, regex.VERBOSE)
