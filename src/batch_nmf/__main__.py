"""
Entry-point module, in case you use `python -m pytorch_nmf`.

Why does this file exist, and why `__main__`? For more info, read:

- https://www.python.org/dev/peps/pep-0338/
- https://docs.python.org/3/using/cmdline.html#cmdoption-m
"""

import sys

from pytorch_nmf.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
