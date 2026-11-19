"""Verify that every published figure in README.md and ADRs matches docs/metrics.json."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "docs" / "metrics.json"


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _format_anchor(template: str, value: int | float) -> str:
    if isinstance(value, float) and value.is_integer() and "{}" in template:
        return template.format(int(value))
    return template.format(value)


def _candidate_phrases(templates: list[str], value: int | float) -> list[str]:
    phrases: list[str] = []
    for tmpl in templates:
        phrases.append(_format_anchor(tmpl, value))
        if isinstance(value, float):
            phrases.append(tmpl.format(f"{value:.1f}"))
    return list(dict.fromkeys(phrases))


def _template_to_regex(template: str) -> re.Pattern[str]:
    placeholder = "__NUM__"
    marked = template.replace("{}", placeholder)
    escaped = re.escape(marked)
    pattern = escaped.replace(placeholder, r"(\d[\d,]*(?:\.\d+)?)")
    return re.compile(pattern)


def main() -> int:
    if not METRICS_PATH.exists():
        print(f"error: {METRICS_PATH} missing; run tools/collect_metrics.py", file=sys.stderr)
        return 1

    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metrics: dict[str, int | float] = payload["metrics"]
    anchors: dict[str, list[str]] = payload["anchors"]
    checked_docs: list[str] = payload["checked_documents"]

    missing_docs = [doc for doc in checked_docs if not (ROOT / doc).exists()]
    if missing_docs:
        print(f"error: checked documents missing: {missing_docs}", file=sys.stderr)
        return 1

    doc_texts = {
        doc: _normalize_whitespace((ROOT / doc).read_text(encoding="utf-8"))
        for doc in checked_docs
    }
    readme_text = doc_texts["README.md"]

    failures: list[str] = []
    for key, value in metrics.items():
        templates = anchors.get(key, [])
        if not templates:
            failures.append(f"metric '{key}' ({value}) has no anchor phrase")
            continue

        phrases = _candidate_phrases(templates, value)
        if not any(p in readme_text for p in phrases):
            failures.append(f"README.md missing metric '{key}'={value} via anchors {phrases}")

        for doc, text in doc_texts.items():
            for tmpl in templates:
                regex = _template_to_regex(tmpl)
                for match in regex.finditer(text):
                    matched_phrase = match.group(0)
                    if matched_phrase not in phrases:
                        failures.append(
                            f"{doc}: stale figure '{matched_phrase}' for metric '{key}' "
                            f"(expected one of {phrases})"
                        )

    if failures:
        print("check_numbers: FAILED", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print(
        f"check_numbers: PASSED ({len(metrics)} metrics verified across "
        f"{len(checked_docs)} documents)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
