"""ASCII-art wordmark used as the page heading.

Kept in its own module so the art stays intact: it only reads correctly if
every leading space survives, and the rows are padded to a uniform width. The
width is imported by app.py to size the text, so the CSS cannot drift out of
step with the art.
"""

WIDTH = 73
HEIGHT = 3

# The first row begins immediately after the opening quotes on purpose: a
# newline there would add a blank line above the art.
LOGO = r"""██▄  ▄██ ▄▄▄▄▄  ▄▄▄  ▄▄      █████▄ ▄▄     ▄▄▄  ▄▄  ▄▄ ▄▄  ▄▄ ▄▄▄▄▄ ▄▄▄▄ 
██ ▀▀ ██ ██▄▄  ██▀██ ██      ██▄▄█▀ ██    ██▀██ ███▄██ ███▄██ ██▄▄  ██▄█▄
██    ██ ██▄▄▄ ██▀██ ██▄▄▄   ██     ██▄▄▄ ██▀██ ██ ▀██ ██ ▀██ ██▄▄▄ ██ ██"""
