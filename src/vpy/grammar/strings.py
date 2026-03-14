import regex


stringprefix = r"""
(
    [rRuUfF]
    |
    [rR][fF]
    |
    [fF][rR]
)
"""
stringprefix_re = regex.compile(stringprefix, regex.VERBOSE)

shortstringchar = r"""
(
    [^\\\n"]
)
"""
shortstringchar_re = regex.compile(shortstringchar, regex.VERBOSE)

longstringchar = r"""
(
    [^\\]
)
"""
longstringchar_re = regex.compile(longstringchar, regex.VERBOSE)

stringescapeseq = r"""
(
    \\
    .
)
"""
stringescapeseq_re = regex.compile(stringescapeseq, regex.VERBOSE)

shortstringitem = rf"""
(
    {shortstringchar}
    |
    {stringescapeseq}
)
"""
shortstringitem_re = regex.compile(shortstringitem, regex.VERBOSE)

longstringitem = rf"""
(
    {longstringchar}
    |
    {stringescapeseq}
)
"""
longstringitem_re = regex.compile(longstringitem, regex.VERBOSE)

shortstring = rf"""
(
    ' {shortstringitem}* '
    |
    " {shortstringitem}* "
)
"""
shortstring_re = regex.compile(shortstring, regex.VERBOSE)

longstring = rf"""
(
    ''' {shortstringitem}* '''
    |
    ""\" {shortstringitem}* ""\"
)
"""
longstring_re = regex.compile(longstring, regex.VERBOSE)

stringliteral = rf"""
(
    {stringprefix}?
    (
        {shortstring}
        |
        {longstring}
    )
)
"""
stringliteral_re = regex.compile(stringliteral, regex.VERBOSE)
