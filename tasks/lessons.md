# Lessons

## Python Version Compatibility — Oracle VM

Oracle VM runs Python 3.9. PEP 604 union syntax (`X | None`) and builtin generics
(`list[X]`, `dict[X, Y]`, `tuple[X, Y]`) in annotations require Python 3.10+.

**Rule:** Every `.py` file must have `from __future__ import annotations` as the first
non-docstring line. This enables PEP 563 deferred annotation evaluation and makes all
modern type hint syntax work on Python 3.7+.

**Pre-deploy check:** Before any deploy, run:
```
grep -rL "from __future__ import annotations" *.py strategies/ config/
```
Any file returned that uses type hints needs the import added.
