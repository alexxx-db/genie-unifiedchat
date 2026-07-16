"""Extract concrete literals from executed warehouse results for Genie follow-ups.

Used on sequential loops: after Query N succeeds, feed IDs/codes/values into the
next Genie question so Genie does not rediscover a set already returned by SQL.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

MAX_ROWS_SCANNED = 100
MAX_LITERALS_PER_COLUMN = 25
MAX_TOTAL_LITERALS = 50
MAX_SAMPLE_ROWS = 8
MAX_CELL_CHARS = 64
MAX_BLOCK_CHARS = 2500

# Columns that usually hold join keys / codes rather than pure metrics.
_CODEISH_COLUMN = re.compile(
    r"(id|code|ndc|npi|sku|key|name|drug|diag|cpt|icd|member|provider|product)",
    re.IGNORECASE,
)
# Columns that are almost always aggregates / measures — skip numeric cells.
_METRICISH_COLUMN = re.compile(
    r"(cost|amount|price|total|sum|count|avg|mean|rate|pct|percent|qty|quantity|score)",
    re.IGNORECASE,
)


def _is_success_result(result: Dict[str, Any]) -> bool:
    status = str(result.get("status") or "").lower()
    if status in {"failed", "skipped", "error"}:
        return False
    if result.get("success") is False:
        return False
    return bool(result.get("result") is not None or result.get("columns"))


def _cell_to_literal(value: Any, *, allow_numeric: bool) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if not allow_numeric:
            return None
        if isinstance(value, float):
            # Prefer integers that look like codes; skip noisy floats.
            if not value.is_integer() or abs(value) >= 1e12:
                return None
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text or len(text) > MAX_CELL_CHARS:
        return None
    # Skip free-form blobs / JSON-looking cells.
    if text.startswith("{") or text.startswith("["):
        return None
    if not allow_numeric and re.fullmatch(r"-?\d+(\.\d+)?", text):
        return None
    return text


def extract_literals_from_rows(
    columns: Sequence[str],
    rows: Sequence[Any],
    *,
    max_per_column: int = MAX_LITERALS_PER_COLUMN,
    max_total: int = MAX_TOTAL_LITERALS,
) -> Dict[str, Any]:
    """Extract per-column unique literals from normalized row dicts."""
    by_column: Dict[str, List[str]] = {}
    all_literals: List[str] = []
    seen_all: Set[str] = set()

    col_names = [str(c) for c in columns if c is not None]
    for col in col_names:
        # Allow numerics for code-ish columns (NPI, ICD codes as ints); skip
        # metric columns so costs/counts don't pollute Genie prompts.
        allow_numeric = bool(_CODEISH_COLUMN.search(col)) and not bool(
            _METRICISH_COLUMN.search(col)
        )
        if _METRICISH_COLUMN.search(col) and not _CODEISH_COLUMN.search(col):
            continue

        values: List[str] = []
        seen_col: Set[str] = set()
        for row in rows[:MAX_ROWS_SCANNED]:
            if not isinstance(row, dict):
                continue
            literal = _cell_to_literal(row.get(col), allow_numeric=allow_numeric)
            if literal is None:
                continue
            key = literal.lower()
            if key in seen_col:
                continue
            seen_col.add(key)
            values.append(literal)
            if key not in seen_all and len(all_literals) < max_total:
                seen_all.add(key)
                all_literals.append(literal)
            if len(values) >= max_per_column:
                break
        if values:
            by_column[col] = values

    # Prefer code-ish columns when ranking "key" literals for Genie prompts.
    preferred: List[str] = []
    preferred_seen: Set[str] = set()
    for col, values in by_column.items():
        if not _CODEISH_COLUMN.search(col):
            continue
        for value in values:
            key = value.lower()
            if key in preferred_seen:
                continue
            preferred_seen.add(key)
            preferred.append(value)
            if len(preferred) >= max_total:
                break
        if len(preferred) >= max_total:
            break

    key_literals = preferred or all_literals
    return {
        "by_column": by_column,
        "literals": key_literals[:max_total],
        "all_literals": all_literals[:max_total],
    }


def build_executed_literal_package(
    execution_results: Optional[Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a compact package from one or more successful execution results."""
    packages: List[Dict[str, Any]] = []
    combined_by_column: Dict[str, List[str]] = {}
    combined_literals: List[str] = []
    seen_literals: Set[str] = set()

    for result in execution_results or []:
        if not isinstance(result, dict) or not _is_success_result(result):
            continue
        columns = list(result.get("columns") or [])
        rows = list(result.get("result") or [])
        extracted = extract_literals_from_rows(columns, rows)
        sample_rows = rows[:MAX_SAMPLE_ROWS]
        entry = {
            "query_number": result.get("query_number"),
            "query_label": result.get("query_label"),
            "row_count": result.get("row_count", len(rows)),
            "columns": columns,
            "sql_preview": str(result.get("sql") or "")[:500],
            "by_column": extracted["by_column"],
            "literals": extracted["literals"],
            "sample_rows": sample_rows,
        }
        packages.append(entry)

        for col, values in extracted["by_column"].items():
            bucket = combined_by_column.setdefault(col, [])
            seen_col = {v.lower() for v in bucket}
            for value in values:
                if value.lower() in seen_col:
                    continue
                seen_col.add(value.lower())
                bucket.append(value)
                if len(bucket) >= MAX_LITERALS_PER_COLUMN:
                    break

        for value in extracted["literals"]:
            key = value.lower()
            if key in seen_literals:
                continue
            seen_literals.add(key)
            combined_literals.append(value)
            if len(combined_literals) >= MAX_TOTAL_LITERALS:
                break

    return {
        "query_count": len(packages),
        "queries": packages,
        "by_column": combined_by_column,
        "literals": combined_literals,
        "has_literals": bool(combined_literals or combined_by_column),
    }


def format_executed_literals_block(package: Optional[Dict[str, Any]]) -> str:
    """Render an inject block for Genie / synthesis prompts."""
    if not package or not package.get("has_literals"):
        return ""

    lines = [
        "EXECUTED RESULT LITERALS "
        "(from prior warehouse query — use these concrete values in the next "
        "Genie question; do NOT rediscover this set):"
    ]

    by_column = package.get("by_column") or {}
    # Show code-ish columns first, then the rest.
    ordered_cols = sorted(
        by_column.keys(),
        key=lambda c: (0 if _CODEISH_COLUMN.search(c) else 1, c.lower()),
    )
    for col in ordered_cols[:12]:
        values = by_column[col][:MAX_LITERALS_PER_COLUMN]
        if not values:
            continue
        rendered = ", ".join(values)
        if len(rendered) > 240:
            rendered = rendered[:237] + "..."
        lines.append(f"- {col}: {rendered}")

    literals = package.get("literals") or []
    if literals:
        rendered = ", ".join(literals[:MAX_TOTAL_LITERALS])
        if len(rendered) > 400:
            rendered = rendered[:397] + "..."
        lines.append(f"- Key IDs / literals: {rendered}")

    # Compact IN-list hint for the most code-ish column.
    for col in ordered_cols:
        if not _CODEISH_COLUMN.search(col):
            continue
        values = by_column.get(col) or []
        if not values:
            continue
        quoted = ", ".join("'" + v.replace("'", "''") + "'" for v in values[:20])
        lines.append(
            f"- Suggested filter hint: {col} IN ({quoted})"
            + (" ..." if len(values) > 20 else "")
        )
        break

    queries = package.get("queries") or []
    if queries:
        latest = queries[-1]
        label = latest.get("query_label") or f"Query {latest.get('query_number', '?')}"
        lines.append(
            f"- Source: {label} ({latest.get('row_count', 0)} rows)"
        )

    block = "\n".join(lines)
    if len(block) > MAX_BLOCK_CHARS:
        return block[: MAX_BLOCK_CHARS - 3].rstrip() + "..."
    return block


def enrich_question_with_executed_literals(
    question: str,
    package: Optional[Dict[str, Any]],
) -> str:
    """Append the executed-literal inject block to a Genie / synthesis question."""
    base = (question or "").strip()
    block = format_executed_literals_block(package)
    if not block:
        return base
    if not base:
        return block
    if "EXECUTED RESULT LITERALS" in base:
        return base
    return f"{base}\n\n{block}"


def package_to_json(package: Optional[Dict[str, Any]], *, max_chars: int = 2000) -> str:
    """Compact JSON for feedback blobs (truncates sample rows)."""
    if not package:
        return "{}"
    slim = {
        "query_count": package.get("query_count", 0),
        "by_column": package.get("by_column") or {},
        "literals": package.get("literals") or [],
        "queries": [
            {
                "query_number": q.get("query_number"),
                "query_label": q.get("query_label"),
                "row_count": q.get("row_count"),
                "columns": q.get("columns"),
                "literals": q.get("literals"),
            }
            for q in (package.get("queries") or [])
        ],
    }
    text = json.dumps(slim, default=str)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text
