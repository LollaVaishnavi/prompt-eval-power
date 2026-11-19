"""Fallback static lint checker used when ruff is not installed locally."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

TARGET_DIRS = ("src", "tests", "experiments", "tools")
MAX_LINE_LENGTH = 100
EXEMPT_LONG_LINE_FILES = {"report.py", "render_diagrams.py"}


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        errors.append(f"{path}:{exc.lineno}: syntax error: {exc.msg}")
        return errors

    for lineno, line in enumerate(source.splitlines(), start=1):
        if "\u2014" in line:
            errors.append(f"{path}:{lineno}: em dash found")
        if line.endswith(" ") or line.endswith("\t"):
            errors.append(f"{path}:{lineno}: trailing whitespace")
        if (
            path.name not in EXEMPT_LONG_LINE_FILES
            and len(line) > MAX_LINE_LENGTH
            and "http://" not in line
            and "https://" not in line
        ):
            errors.append(f"{path}:{lineno}: line too long ({len(line)} > {MAX_LINE_LENGTH})")
    return errors


def main() -> int:
    all_errors: list[str] = []
    for folder in TARGET_DIRS:
        root = Path(folder)
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            all_errors.extend(check_file(py_file))

    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        return 1

    print("lint_check: all Python files clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
