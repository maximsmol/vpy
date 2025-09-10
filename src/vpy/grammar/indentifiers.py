from textwrap import dedent

import regex

# todo(maximsmol): deal with NFKC normalization
# todo(maximsmol): use the hardcoded list of characters for Other_ID_Start and Other_ID_Continue?
id_start = r"""
\p{Lu}
\p{Ll}
\p{Lt}
\p{Lm}
\p{Lo}
\p{Nl}
\_
\p{Other_ID_Start}
""".replace("\n", "")

id_continue = id_start + dedent(r"""
    \p{Mn}
    \p{Mc}
    \p{Nd}
    \p{Pc}
    \p{Other_ID_Continue}
    """).replace("\n", "")

identifier = rf"""
(
    [{id_start}]
    [{id_continue}]*
)
"""

identifier_re = regex.compile(identifier, regex.VERBOSE)
