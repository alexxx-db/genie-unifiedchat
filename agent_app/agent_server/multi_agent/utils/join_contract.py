"""Shared join-contract artifact for multi-space Genie / SQL chaining.

Normalized shape kept in agent state and passed forward to downstream prompts::

    {
      "version": 1,
      "entities": ["Pharmacy Claims", ...],
      "keys": ["D001", "D002"],
      "key_columns": ["drug_code"],
      "time_window": {"start": "2024-01-01", "end": "2024-12-31", "raw": [...]},
      "metrics": ["total_cost"],
      "sql_by_space": {"space_a": "SELECT ..."},
      "conversation_ids": {"space_a": "conv-..."},
      "dependency_edges": [{"from": "space_a", "to": "space_b"}],
      "literals_by_column": {"drug_code": ["D001", "D002"]},
      "source": "merged",
      "notes": []
    }
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .executed_result_literals import (
    MAX_LITERALS_PER_COLUMN,
    MAX_TOTAL_LITERALS,
    build_executed_literal_package,
)
from .genie_route_dag import (
    build_context_package,
    extract_conversation_ids,
    normalize_genie_route_plan,
)

JOIN_CONTRACT_VERSION = 1
MAX_ENTITIES = 20
MAX_METRICS = 20
MAX_SQL_PREVIEW = 600
MAX_BLOCK_CHARS = 3000

_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2}|20\d{2})\b")
_METRIC_RE = re.compile(
    r"(cost|amount|price|total|sum|count|avg|mean|rate|pct|percent|qty|quantity)",
    re.IGNORECASE,
)
_CODEISH_COLUMN = re.compile(
    r"(id|code|ndc|npi|sku|key|name|drug|diag|cpt|icd|member|provider|product)",
    re.IGNORECASE,
)


def empty_join_contract(*, source: str = "empty") -> Dict[str, Any]:
    return {
        "version": JOIN_CONTRACT_VERSION,
        "entities": [],
        "keys": [],
        "key_columns": [],
        "time_window": {"start": None, "end": None, "raw": []},
        "metrics": [],
        "sql_by_space": {},
        "conversation_ids": {},
        "dependency_edges": [],
        "literals_by_column": {},
        "source": source,
        "notes": [],
    }


def _dedupe_preserve(items: Iterable[Any], *, limit: Optional[int] = None) -> List[Any]:
    out: List[Any] = []
    seen: Set[str] = set()
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            key = repr(sorted(item.items()))
            value = item
        else:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            value = text if not isinstance(item, (int, float, bool)) else item
            if isinstance(item, str):
                value = text
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if limit is not None and len(out) >= limit:
            break
    return out


def _merge_str_lists(*lists: Optional[Sequence[Any]], limit: int) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        merged.extend(str(x).strip() for x in lst if str(x).strip())
    return _dedupe_preserve(merged, limit=limit)


def _merge_column_maps(
    *maps: Optional[Dict[str, Sequence[str]]],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for mapping in maps:
        if not mapping:
            continue
        for col, values in mapping.items():
            col_name = str(col).strip()
            if not col_name:
                continue
            bucket = out.setdefault(col_name, [])
            for value in _merge_str_lists(bucket, values, limit=MAX_LITERALS_PER_COLUMN):
                if value not in bucket:
                    bucket.append(value)
                if len(bucket) >= MAX_LITERALS_PER_COLUMN:
                    break
    return out


def _extract_time_window_from_texts(texts: Sequence[str]) -> Dict[str, Any]:
    raw: List[str] = []
    for text in texts:
        if not text:
            continue
        for match in _DATE_RE.findall(str(text)):
            raw.append(match)
    raw = _dedupe_preserve(raw, limit=12)
    start = raw[0] if raw else None
    end = raw[-1] if len(raw) > 1 else start
    # Prefer lexicographic min/max for ISO-ish dates/years.
    if raw:
        ordered = sorted(raw)
        start, end = ordered[0], ordered[-1]
    return {"start": start, "end": end, "raw": raw}


def _extract_metrics_from_columns(columns: Sequence[str]) -> List[str]:
    metrics = [c for c in columns if c and _METRIC_RE.search(str(c))]
    return _dedupe_preserve(metrics, limit=MAX_METRICS)


def merge_join_contracts(*contracts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge contracts left-to-right; later non-empty fields win / extend."""
    base = empty_join_contract(source="merged")
    sources: List[str] = []
    notes: List[str] = []

    for contract in contracts:
        if not contract or not isinstance(contract, dict):
            continue
        sources.append(str(contract.get("source") or "unknown"))
        notes.extend(contract.get("notes") or [])

        base["entities"] = _merge_str_lists(
            base["entities"], contract.get("entities"), limit=MAX_ENTITIES
        )
        base["keys"] = _merge_str_lists(
            base["keys"], contract.get("keys"), limit=MAX_TOTAL_LITERALS
        )
        base["key_columns"] = _merge_str_lists(
            base["key_columns"], contract.get("key_columns"), limit=MAX_ENTITIES
        )
        base["metrics"] = _merge_str_lists(
            base["metrics"], contract.get("metrics"), limit=MAX_METRICS
        )
        base["literals_by_column"] = _merge_column_maps(
            base["literals_by_column"],
            contract.get("literals_by_column"),
        )

        sql_by_space = dict(base["sql_by_space"])
        for sid, sql in (contract.get("sql_by_space") or {}).items():
            text = str(sql or "").strip()
            if text:
                sql_by_space[str(sid)] = text[:MAX_SQL_PREVIEW]
        base["sql_by_space"] = sql_by_space

        cids = dict(base["conversation_ids"])
        for sid, cid in (contract.get("conversation_ids") or {}).items():
            value = str(cid or "").strip()
            if value:
                cids[str(sid)] = value
        base["conversation_ids"] = cids

        edges = list(base["dependency_edges"])
        seen_edges = {(e.get("from"), e.get("to")) for e in edges if isinstance(e, dict)}
        for edge in contract.get("dependency_edges") or []:
            if not isinstance(edge, dict):
                continue
            key = (edge.get("from"), edge.get("to"))
            if not key[0] or not key[1] or key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({"from": key[0], "to": key[1]})
        base["dependency_edges"] = edges

        tw = contract.get("time_window") or {}
        merged_raw = _merge_str_lists(
            (base.get("time_window") or {}).get("raw"),
            tw.get("raw"),
            limit=12,
        )
        if tw.get("start") or tw.get("end") or merged_raw:
            ordered = sorted(merged_raw) if merged_raw else []
            base["time_window"] = {
                "start": tw.get("start") or (ordered[0] if ordered else base["time_window"].get("start")),
                "end": tw.get("end") or (ordered[-1] if ordered else base["time_window"].get("end")),
                "raw": ordered or merged_raw,
            }

    if sources:
        base["source"] = "merged" if len(set(sources)) > 1 else sources[-1]
    base["notes"] = _dedupe_preserve(notes, limit=20)

    # If keys empty but column map has code-ish values, promote them.
    if not base["keys"] and base["literals_by_column"]:
        promoted: List[str] = []
        for col, values in base["literals_by_column"].items():
            if _CODEISH_COLUMN.search(col):
                promoted.extend(values)
        base["keys"] = _dedupe_preserve(promoted, limit=MAX_TOTAL_LITERALS)
        base["key_columns"] = _merge_str_lists(
            base["key_columns"],
            [c for c in base["literals_by_column"] if _CODEISH_COLUMN.search(c)],
            limit=MAX_ENTITIES,
        )

    return base


def build_join_contract_from_plan(
    plan: Optional[Dict[str, Any]],
    *,
    relevant_spaces: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Seed a join contract from the planner output (before Genie/SQL runs)."""
    if not isinstance(plan, dict):
        return empty_join_contract(source="planning")

    contract = empty_join_contract(source="planning")
    space_titles = []
    for space in relevant_spaces or plan.get("vector_search_relevant_spaces_info") or []:
        if isinstance(space, dict):
            title = space.get("space_title") or space.get("space_id")
            if title:
                space_titles.append(str(title))
    contract["entities"] = _dedupe_preserve(space_titles, limit=MAX_ENTITIES)
    contract["dependency_edges"] = list(plan.get("dependency_edges") or [])

    normalized = normalize_genie_route_plan(plan.get("genie_route_plan"))
    texts = [str(plan.get("original_query") or ""), str(plan.get("execution_plan") or "")]
    for step in normalized.values():
        texts.append(str(step.get("question") or ""))
    contract["time_window"] = _extract_time_window_from_texts(texts)

    if plan.get("requires_join"):
        contract["notes"] = ["Planner marked requires_join=true"]
    return contract


def build_join_contract_from_genie_results(
    genie_results: Optional[Dict[str, Any]],
    *,
    relevant_spaces: Optional[Sequence[Dict[str, Any]]] = None,
    dependency_edges: Optional[Sequence[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Build/update contract fields from Genie tool outputs."""
    contract = empty_join_contract(source="genie")
    if not genie_results or not isinstance(genie_results, dict):
        return contract

    title_by_id = {
        str(s.get("space_id")): str(s.get("space_title") or s.get("space_id"))
        for s in (relevant_spaces or [])
        if isinstance(s, dict) and s.get("space_id")
    }

    sql_by_space: Dict[str, str] = {}
    keys: List[str] = []
    texts: List[str] = []
    entities: List[str] = []

    for space_id, payload in genie_results.items():
        if str(space_id).startswith("_") or not isinstance(payload, dict):
            continue
        sid = str(space_id)
        entities.append(title_by_id.get(sid, sid))
        sql = str(payload.get("sql") or "").strip()
        answer = str(payload.get("answer") or "").strip()
        if sql:
            sql_by_space[sid] = sql[:MAX_SQL_PREVIEW]
            texts.append(sql)
        if answer:
            texts.append(answer)
        pkg = build_context_package(sid, payload)
        keys.extend(pkg.get("ids") or [])
        for predicate in pkg.get("filters") or []:
            texts.append(str(predicate))

    contract["entities"] = _dedupe_preserve(entities, limit=MAX_ENTITIES)
    contract["keys"] = _dedupe_preserve(keys, limit=MAX_TOTAL_LITERALS)
    contract["sql_by_space"] = sql_by_space
    contract["conversation_ids"] = extract_conversation_ids(genie_results)
    contract["dependency_edges"] = list(dependency_edges or [])
    if isinstance(genie_results.get("_genie_dag"), dict):
        dag_deps = (genie_results["_genie_dag"] or {}).get("dependencies") or {}
        if dag_deps and not contract["dependency_edges"]:
            edges = []
            for dst, srcs in dag_deps.items():
                for src in srcs or []:
                    edges.append({"from": src, "to": dst})
            contract["dependency_edges"] = edges
    contract["time_window"] = _extract_time_window_from_texts(texts)
    return contract


def build_join_contract_from_execution(
    execution_results: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    literal_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build contract fields from warehouse execution results / literal package."""
    package = literal_package or build_executed_literal_package(execution_results)
    contract = empty_join_contract(source="execution")
    if not package or not package.get("has_literals"):
        # Still capture metrics/SQL from results even without literals.
        pass

    contract["keys"] = list(package.get("literals") or [])[:MAX_TOTAL_LITERALS]
    contract["literals_by_column"] = dict(package.get("by_column") or {})
    contract["key_columns"] = [
        c for c in contract["literals_by_column"] if _CODEISH_COLUMN.search(c)
    ][:MAX_ENTITIES]

    metrics: List[str] = []
    texts: List[str] = []
    sql_by_space: Dict[str, str] = {}
    for idx, result in enumerate(execution_results or []):
        if not isinstance(result, dict):
            continue
        cols = [str(c) for c in (result.get("columns") or [])]
        metrics.extend(_extract_metrics_from_columns(cols))
        sql = str(result.get("sql") or "").strip()
        if sql:
            texts.append(sql)
            label = str(result.get("query_label") or f"query_{result.get('query_number', idx + 1)}")
            sql_by_space[label] = sql[:MAX_SQL_PREVIEW]
    contract["metrics"] = _dedupe_preserve(metrics, limit=MAX_METRICS)
    contract["sql_by_space"] = sql_by_space
    contract["time_window"] = _extract_time_window_from_texts(
        texts + [str(v) for values in contract["literals_by_column"].values() for v in values]
    )
    if package.get("has_literals"):
        contract["notes"] = ["Merged executed warehouse literals"]
    return contract


def format_join_contract_block(contract: Optional[Dict[str, Any]]) -> str:
    """Render a compact prompt block for Genie / synthesis follow-ups."""
    if not contract or not isinstance(contract, dict):
        return ""

    # Skip empty contracts.
    meaningful = any(
        [
            contract.get("entities"),
            contract.get("keys"),
            contract.get("key_columns"),
            contract.get("metrics"),
            contract.get("sql_by_space"),
            (contract.get("time_window") or {}).get("raw"),
            contract.get("dependency_edges"),
            contract.get("literals_by_column"),
        ]
    )
    if not meaningful:
        return ""

    lines = [
        "JOIN CONTRACT (shared cross-space context — use these fields in downstream "
        "Genie questions / SQL; do not rediscover):"
    ]
    if contract.get("entities"):
        lines.append(f"- Entities / spaces: {', '.join(contract['entities'][:MAX_ENTITIES])}")
    if contract.get("key_columns"):
        lines.append(f"- Key columns: {', '.join(contract['key_columns'][:MAX_ENTITIES])}")
    if contract.get("keys"):
        rendered = ", ".join(str(k) for k in contract["keys"][:MAX_TOTAL_LITERALS])
        if len(rendered) > 400:
            rendered = rendered[:397] + "..."
        lines.append(f"- Keys / literals: {rendered}")
    by_col = contract.get("literals_by_column") or {}
    for col in list(by_col.keys())[:8]:
        vals = ", ".join(str(v) for v in (by_col[col] or [])[:MAX_LITERALS_PER_COLUMN])
        if len(vals) > 200:
            vals = vals[:197] + "..."
        lines.append(f"- {col}: {vals}")
    tw = contract.get("time_window") or {}
    if tw.get("start") or tw.get("end") or tw.get("raw"):
        if tw.get("start") and tw.get("end") and tw.get("start") != tw.get("end"):
            lines.append(f"- Time window: {tw['start']} → {tw['end']}")
        elif tw.get("start") or tw.get("end"):
            lines.append(f"- Time window: {tw.get('start') or tw.get('end')}")
        elif tw.get("raw"):
            lines.append(f"- Time markers: {', '.join(tw['raw'][:8])}")
    if contract.get("metrics"):
        lines.append(f"- Metrics: {', '.join(contract['metrics'][:MAX_METRICS])}")
    edges = contract.get("dependency_edges") or []
    if edges:
        edge_txt = ", ".join(f"{e.get('from')}→{e.get('to')}" for e in edges[:12] if isinstance(e, dict))
        if edge_txt:
            lines.append(f"- Dependency edges: {edge_txt}")
    sql_by_space = contract.get("sql_by_space") or {}
    for sid, sql in list(sql_by_space.items())[:4]:
        preview = str(sql).replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        lines.append(f"- SQL[{sid}]: {preview}")
    cids = contract.get("conversation_ids") or {}
    if cids:
        lines.append(
            "- Genie conversation_ids: "
            + ", ".join(f"{sid}={cid}" for sid, cid in list(cids.items())[:6])
        )

    block = "\n".join(lines)
    if len(block) > MAX_BLOCK_CHARS:
        return block[: MAX_BLOCK_CHARS - 3].rstrip() + "..."
    return block


def enrich_question_with_join_contract(
    question: str,
    contract: Optional[Dict[str, Any]],
) -> str:
    """Append the join-contract block to a Genie / synthesis question."""
    base = (question or "").strip()
    block = format_join_contract_block(contract)
    if not block:
        return base
    if not base:
        return block
    if "JOIN CONTRACT" in base:
        return base
    return f"{base}\n\n{block}"


def join_contract_summary(contract: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact summary for logging / plan_formulation events."""
    if not contract:
        return {"present": False}
    return {
        "present": True,
        "source": contract.get("source"),
        "entity_count": len(contract.get("entities") or []),
        "key_count": len(contract.get("keys") or []),
        "metric_count": len(contract.get("metrics") or []),
        "sql_space_count": len(contract.get("sql_by_space") or {}),
        "edge_count": len(contract.get("dependency_edges") or []),
        "has_time_window": bool((contract.get("time_window") or {}).get("raw")),
    }
