import regex

from .integers import digit

digitpart = rf"""
(
    {digit}
    (
        _?
        {digit}
    )*
)
"""
digitpart_re = regex.compile(digitpart, regex.VERBOSE)

fraction = rf"""
(
    \.
    {digitpart}
)
"""
fraction_re = regex.compile(digitpart, regex.VERBOSE)


exponent = rf"""
(
    [eE]
    [+-]?
    {digitpart}
)
"""
exponent_re = regex.compile(digitpart, regex.VERBOSE)

pointfloat = rf"""
(
    {digitpart}?
    {fraction}
    |
    {digitpart}
    \.
)
"""
pointfloat_re = regex.compile(pointfloat, regex.VERBOSE)

exponentfloat = rf"""
(
    (
        {digitpart}
        |
        {pointfloat}
    )
    {exponent}
)
"""
exponentfloat_re = regex.compile(pointfloat, regex.VERBOSE)

floatnumber = rf"""
(
    {pointfloat}
    |
    {exponentfloat}
)
"""
floatnumber_re = regex.compile(pointfloat, regex.VERBOSE)
