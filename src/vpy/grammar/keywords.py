import re

keywords = r"""
(
  False
| None
| True
| and
| as
| assert
| async
| await
| break
| class
| continue
| def
| del
| elif
| else
| except
| finally
| for
| from
| global
| if
| import
| in
| is
| lambda
| nonlocal
| not
| or
| pass
| raise
| return
| try
| while
| with
| yield
)
"""

keywords_re = re.compile(keywords, re.VERBOSE)
