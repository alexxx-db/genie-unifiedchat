"""Genie route DAG helpers: normalize plans, compute waves, inject upstream context.

Supports legacy ``{space_id: question_str}`` plans and structured steps::

    {
      "space_a": {
        "question": "...",
        "depends_on": [],
        "inject": ["filters", "ids", "sql_preview", "answer_summary"]
      },
      "space_b": {
        "question": "...",
        "depends_on": ["space_a"],
        "inject": ["filters", "ids", "sql_preview", "answer_summary"]
      }
    }

``genie_execution_mode`` is separate from the UI SQL ``execution_mode``
(parallel/sequential query execution). Values: ``"parallel"`` | ``"dag"``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

VALID_INJECT_FIELDS = ("filters", "ids", "sql_preview", "answer_summary")
DEFAULT_INJECT_FIELDS = list(VALID_INJECT_FIELDS)
MAX_ANSWER_SUMMARY_CHARS = 600
MAX_SQL_PREVIEW_CHARS = 800
MAX_IDS = 25
MAX_FILTERS = 15


def normalize_genie_route_step(space_id: str, raw: Any) -> Dict[str, Any]:
    """Normalize one plan entry to a structured step dict."""
    if isinstance(raw, str):
        return {
            "space_id": space_id,
            "question": raw.strip(),
            "depends_on": [],
            "inject": list(DEFAULT_INJECT_FIELDS),
        }

    if not isinstance(raw, dict):
        return {
            "space_id": space_id,
            "question": str(raw).strip(),
            "depends_on": [],
            "inject": list(DEFAULT_INJECT_FIELDS),
        }

    question = (
        raw.get("question")
        or raw.get("partial_question")
        or raw.get("query")
        or ""
    )
    depends_on = raw.get("depends_on") or raw.get("dependencies") or []
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    depends_on = [str(d).strip() for d in depends_on if str(d).strip()]

    inject = raw.get("inject")
    if inject is None:
        inject = list(DEFAULT_INJECT_FIELDS)
    elif isinstance(inject, str):
        inject = [inject]
    inject = [str(f).strip() for f in inject if str(f).strip() in VALID_INJECT_FIELDS]

    return {
        "space_id": space_id,
        "question": str(question).strip(),
        "depends_on": depends_on,
        "inject": inject or list(DEFAULT_INJECT_FIELDS),
    }


def normalize_genie_route_plan(
    genie_route_plan: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Normalize a full plan to ``{space_id: structured_step}``."""
    if not genie_route_plan or not isinstance(genie_route_plan, dict):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for space_id, raw in genie_route_plan.items():
        sid = str(space_id).strip()
        if not sid:
            continue
        step = normalize_genie_route_step(sid, raw)
        # Drop self-dependencies and unknown deps later during wave compute.
        step["depends_on"] = [d for d in step["depends_on"] if d != sid]
        if step["question"]:
            normalized[sid] = step
    return normalized


def resolve_genie_execution_mode(
    explicit_mode: Optional[str],
    normalized_plan: Dict[str, Dict[str, Any]],
) -> str:
    """Return ``dag`` or ``parallel``.

    Explicit ``dag`` wins. Otherwise any non-empty ``depends_on`` implies DAG.
    Unknown / empty modes default to parallel when there are no dependencies.
    """
    mode = (explicit_mode or "").strip().lower()
    has_deps = any(step.get("depends_on") for step in normalized_plan.values())
    if mode == "dag" or has_deps:
        return "dag"
    return "parallel"


def compute_execution_waves(
    normalized_plan: Dict[str, Dict[str, Any]],
) -> List[List[str]]:
    """Kahn topological waves. Unknown deps are ignored; cycles become a final wave."""
    if not normalized_plan:
        return []

    space_ids = set(normalized_plan.keys())
    deps: Dict[str, Set[str]] = {}
    dependents: Dict[str, Set[str]] = {sid: set() for sid in space_ids}

    for sid, step in normalized_plan.items():
        valid_deps = {d for d in step.get("depends_on", []) if d in space_ids}
        deps[sid] = valid_deps
        for d in valid_deps:
            dependents[d].add(sid)

    ready = sorted(sid for sid, d in deps.items() if not d)
    remaining = set(space_ids)
    waves: List[List[str]] = []

    while ready:
        wave = list(ready)
        waves.append(wave)
        next_ready: List[str] = []
        for sid in wave:
            remaining.discard(sid)
            for child in sorted(dependents[sid]):
                if child not in remaining:
                    continue
                deps[child].discard(sid)
                if not deps[child] and child not in next_ready:
                    next_ready.append(child)
        ready = sorted(next_ready)

    if remaining:
        # Cycle or unresolved — run leftovers as one last wave without inject order.
        waves.append(sorted(remaining))

    return waves


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _extract_ids(text: str) -> List[str]:
    """Pull quoted strings and code-like tokens useful as join keys / filters."""
    if not text:
        return []
    found: List[str] = []
    seen: Set[str] = set()

    for match in re.findall(r"'([^']{1,64})'|\"([^\"]{1,64})\"", text):
        token = (match[0] or match[1] or "").strip()
        if not token or token.lower() in seen:
            continue
        seen.add(token.lower())
        found.append(token)
        if len(found) >= MAX_IDS:
            return found

    for token in re.findall(r"\b([A-Za-z]{0,4}\d{2,}[A-Za-z0-9._-]{0,20})\b", text):
        if token.lower() in seen:
            continue
        seen.add(token.lower())
        found.append(token)
        if len(found) >= MAX_IDS:
            break

    return found


def _extract_filters(sql: str) -> List[str]:
    """Best-effort extraction of simple comparison predicates from SQL."""
    if not sql:
        return []
    filters: List[str] = []
    seen: Set[str] = set()
    pattern = re.compile(
        r"([A-Za-z_][\w.]*)\s*(=|!=|<>|>=|<=|>|<|IN|LIKE|ILIKE)\s*"
        r"(\((?:[^()]|\([^()]*\))*\)|'[^']*'|\"[^\"]*\"|[^\s,;)]+)",
        re.IGNORECASE,
    )
    for col, op, val in pattern.findall(sql):
        if col.lower() in {"select", "where", "and", "or", "on", "join", "from"}:
            continue
        predicate = f"{col} {op.upper()} {val.strip()}"
        key = predicate.lower()
        if key in seen:
            continue
        seen.add(key)
        filters.append(predicate)
        if len(filters) >= MAX_FILTERS:
            break
    return filters


def build_context_package(space_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact upstream context package from one Genie result."""
    answer = str(result.get("answer") or "")
    sql = str(result.get("sql") or "")
    success = bool(result.get("success") or sql or answer)
    return {
        "space_id": space_id,
        "success": success,
        "answer_summary": _truncate(answer, MAX_ANSWER_SUMMARY_CHARS),
        "sql_preview": _truncate(sql, MAX_SQL_PREVIEW_CHARS),
        "ids": _extract_ids(f"{answer}\n{sql}"),
        "filters": _extract_filters(sql),
        "question": str(result.get("question") or ""),
        "error": str(result.get("error") or ""),
    }


def build_context_packages(
    results_by_space: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {
        sid: build_context_package(sid, res if isinstance(res, dict) else {"answer": str(res)})
        for sid, res in results_by_space.items()
    }


def format_inject_block(
    step: Dict[str, Any],
    packages: Dict[str, Dict[str, Any]],
) -> str:
    """Render upstream context for injection into a dependent Genie question."""
    depends_on: Sequence[str] = step.get("depends_on") or []
    inject_fields: Sequence[str] = step.get("inject") or DEFAULT_INJECT_FIELDS
    if not depends_on:
        return ""

    sections: List[str] = []
    for dep in depends_on:
        pkg = packages.get(dep)
        if not pkg:
            sections.append(f"### From {dep}\n- (no upstream result available)")
            continue

        lines = [f"### From {dep}"]
        if not pkg.get("success"):
            err = pkg.get("error") or "upstream call failed or returned empty"
            lines.append(f"- Upstream status: FAILED ({err})")
        if "answer_summary" in inject_fields and pkg.get("answer_summary"):
            lines.append(f"- Answer summary: {pkg['answer_summary']}")
        if "sql_preview" in inject_fields and pkg.get("sql_preview"):
            lines.append(f"- SQL preview:\n```sql\n{pkg['sql_preview']}\n```")
        if "ids" in inject_fields and pkg.get("ids"):
            lines.append(f"- Key IDs / literals: {', '.join(pkg['ids'])}")
        if "filters" in inject_fields and pkg.get("filters"):
            lines.append(f"- Suggested filters: {'; '.join(pkg['filters'])}")
        if len(lines) == 1:
            lines.append("- (upstream succeeded but no injectable fields)")
        sections.append("\n".join(lines))

    if not sections:
        return ""

    return (
        "CONTEXT FROM UPSTREAM GENIE SPACES "
        "(use these concrete values; do not rediscover them):\n"
        + "\n\n".join(sections)
    )


def enrich_question_with_context(
    step: Dict[str, Any],
    packages: Dict[str, Dict[str, Any]],
) -> str:
    """Return the Genie question, optionally with a structured inject block."""
    question = (step.get("question") or "").strip()
    inject_block = format_inject_block(step, packages)
    if not inject_block:
        return question
    return f"{question}\n\n{inject_block}"


def questions_for_wave(
    wave: Iterable[str],
    normalized_plan: Dict[str, Dict[str, Any]],
    packages: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """Map space_id → enriched question for one execution wave."""
    out: Dict[str, str] = {}
    for sid in wave:
        step = normalized_plan[sid]
        out[sid] = enrich_question_with_context(step, packages)
    return out


def legacy_question_map(normalized_plan: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Flat ``{space_id: base_question}`` without inject (for progress UI / parallel)."""
    return {sid: step["question"] for sid, step in normalized_plan.items()}


def summarize_plan_for_logging(
    normalized_plan: Dict[str, Dict[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    waves = compute_execution_waves(normalized_plan)
    return {
        "genie_execution_mode": mode,
        "space_count": len(normalized_plan),
        "wave_count": len(waves),
        "waves": waves,
        "dependencies": {
            sid: list(step.get("depends_on") or [])
            for sid, step in normalized_plan.items()
        },
        "dependency_edges": dependency_edges_from_plan(normalized_plan),
    }


# ---------------------------------------------------------------------------
# Planner finalize: execution_mode + dependency_edges (P0)
# ---------------------------------------------------------------------------

# Patterns where later Genie questions typically need concrete values from
# earlier ones (top-N → details, find codes → lookup, etc.).
_STAGED_DEPENDENCY_PATTERNS = (
    r"\band their\b",
    r"\band its\b",
    r"\bfor those\b",
    r"\bfor these\b",
    r"\bfor each of (them|those|these)\b",
    r"\bthen\b",
    r"\bfollowed by\b",
    r"\bbased on (those|these|that|the (?:top|above|prior))\b",
    r"\busing (those|these|that)\b",
    r"\btop\s+\d+\b.+\b(and|,)\b",
    r"\b(highest|lowest|most|least)\b.+\b(then|and their|for those)\b",
    r"\b(find|identify|get)\b.+\b(look\s*up|lookup|describe|map)\b",
)


def query_suggests_staged_dependencies(
    query: Optional[str] = None,
    sub_questions: Optional[Sequence[str]] = None,
) -> bool:
    """Heuristic: does the natural-language ask imply staged Genie dependencies?"""
    parts: List[str] = []
    if query:
        parts.append(str(query))
    if sub_questions:
        parts.extend(str(q) for q in sub_questions if q)
    text = " ".join(parts).strip().lower()
    if not text:
        return False
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in _STAGED_DEPENDENCY_PATTERNS)


def dependency_edges_from_plan(
    normalized_plan: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Derive ``[{from, to}, ...]`` edges from structured ``depends_on``."""
    edges: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for sid, step in normalized_plan.items():
        for dep in step.get("depends_on") or []:
            key = (str(dep), str(sid))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": key[0], "to": key[1]})
    return edges


def normalize_dependency_edges(
    raw_edges: Any,
    known_space_ids: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    """Normalize planner-emitted dependency_edges; drop invalid / self edges."""
    if not raw_edges:
        return []
    if not isinstance(raw_edges, list):
        return []

    known = known_space_ids
    edges: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for item in raw_edges:
        if not isinstance(item, dict):
            continue
        src = str(item.get("from") or item.get("source") or "").strip()
        dst = str(item.get("to") or item.get("target") or "").strip()
        if not src or not dst or src == dst:
            continue
        if known is not None and (src not in known or dst not in known):
            continue
        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"from": src, "to": dst})
    return edges


def apply_dependency_edges(
    normalized_plan: Dict[str, Dict[str, Any]],
    edges: Sequence[Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    """Merge explicit edges into each step's ``depends_on`` (deduped)."""
    if not edges:
        return normalized_plan

    updated = {
        sid: {
            **step,
            "depends_on": list(step.get("depends_on") or []),
        }
        for sid, step in normalized_plan.items()
    }
    for edge in edges:
        src = edge["from"]
        dst = edge["to"]
        if dst not in updated or src not in updated:
            continue
        deps = updated[dst]["depends_on"]
        if src not in deps:
            deps.append(src)
    return updated


def ensure_linear_dependencies(
    normalized_plan: Dict[str, Dict[str, Any]],
    space_order: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """If there are no deps, chain spaces in order: each depends on the previous."""
    if not normalized_plan or any(step.get("depends_on") for step in normalized_plan.values()):
        return normalized_plan

    if space_order:
        ordered = [sid for sid in space_order if sid in normalized_plan]
        # Append any spaces missing from the preferred order.
        ordered.extend(sid for sid in normalized_plan if sid not in ordered)
    else:
        ordered = list(normalized_plan.keys())

    if len(ordered) < 2:
        return normalized_plan

    updated = {
        sid: {
            **step,
            "depends_on": list(step.get("depends_on") or []),
            "inject": list(step.get("inject") or DEFAULT_INJECT_FIELDS),
        }
        for sid, step in normalized_plan.items()
    }
    for prev, curr in zip(ordered, ordered[1:]):
        updated[curr]["depends_on"] = [prev]
    return updated


def extract_conversation_ids(
    results: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Pull non-empty per-space Genie conversation_ids from a result map."""
    if not results or not isinstance(results, dict):
        return {}
    out: Dict[str, str] = {}
    for space_id, payload in results.items():
        if str(space_id).startswith("_"):
            continue
        if not isinstance(payload, dict):
            continue
        cid = str(payload.get("conversation_id") or "").strip()
        if cid:
            out[str(space_id)] = cid
    return out


def merge_conversation_ids(
    *maps: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Left-to-right merge; later non-empty values win."""
    merged: Dict[str, str] = {}
    for mapping in maps:
        if not mapping or not isinstance(mapping, dict):
            continue
        for space_id, cid in mapping.items():
            value = str(cid or "").strip()
            if value:
                merged[str(space_id)] = value
    return merged


def resolve_space_conversation_id(
    space_id: str,
    *,
    explicit: Optional[str] = None,
    cached: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Prefer an explicit conversation_id, else a cached per-space id."""
    for candidate in (explicit, (cached or {}).get(space_id)):
        value = str(candidate or "").strip()
        if value:
            return value
    return None


def finalize_planner_genie_plan(plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Finalize planner Genie fields: mode, structured route plan, dependency_edges.

    Idempotent. Clears Genie fields when join_strategy is not genie_route.
    Applies staged-query heuristics when the LLM omitted depends_on / edges.
    """
    if not isinstance(plan, dict):
        return {}

    out = dict(plan)
    join_strategy = (out.get("join_strategy") or "").strip().lower()

    if join_strategy != "genie_route":
        out["genie_route_plan"] = None
        out["genie_execution_mode"] = None
        out["dependency_edges"] = []
        return out

    raw_plan = out.get("genie_route_plan")
    normalized = normalize_genie_route_plan(raw_plan)
    if not normalized:
        out["genie_route_plan"] = None
        out["genie_execution_mode"] = None
        out["dependency_edges"] = []
        return out

    # Merge planner-emitted edges into depends_on.
    explicit_edges = normalize_dependency_edges(
        out.get("dependency_edges"),
        known_space_ids=set(normalized.keys()),
    )
    normalized = apply_dependency_edges(normalized, explicit_edges)

    staged = query_suggests_staged_dependencies(
        query=out.get("original_query") or out.get("execution_plan"),
        sub_questions=out.get("sub_questions"),
    )
    has_deps = any(step.get("depends_on") for step in normalized.values())

    # If the ask is clearly staged but the LLM left a flat parallel plan,
    # materialize a linear dependency chain in relevant_space_ids order.
    if staged and not has_deps and len(normalized) >= 2:
        space_order = out.get("relevant_space_ids") or list(normalized.keys())
        normalized = ensure_linear_dependencies(normalized, space_order)
        has_deps = True

    mode = resolve_genie_execution_mode(out.get("genie_execution_mode"), normalized)
    if staged and len(normalized) >= 2:
        mode = "dag"

    edges = dependency_edges_from_plan(normalized)
    out["genie_route_plan"] = normalized
    out["genie_execution_mode"] = mode
    out["dependency_edges"] = edges
    return out
