"""
Code enrichment utility.

Provides LLM-powered code column detection and batch description lookup.
A single LLM call annotates all unique coded values (NDC, ICD-10, CPT,
NAICS, tickers, etc.) at once — faster and more reliable than web scraping.
"""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


_MAX_CODES_TO_LOOKUP = 30
_AGGREGATE_COLUMN_TOKENS = (
    "count",
    "distinct",
    "unique",
    "total",
    "avg",
    "average",
    "amount",
    "cost",
    "price",
    "sum",
    "rows",
    "days",
    "units",
    "quantity",
    "rank",
    "score",
    "ratio",
    "percent",
    "pct",
)
_CODE_HINT_TOKENS = (
    "code",
    "id",
    "ndc",
    "icd",
    "cpt",
    "hcpcs",
    "npi",
    "taxonomy",
    "zip",
    "state",
    "qual",
    "type",
    "ticker",
    "symbol",
    "fips",
    "cusip",
    "isin",
    "naics",
    "sic",
    "mcc",
)

import ssl
import urllib.request

# Use a default TLS context with certificate + hostname verification enabled.
# These lookups hit public NLM/RxNav HTTPS endpoints, so there is no reason to
# disable verification (doing so exposed responses to MITM tampering, and the
# results are rendered into user-facing output).
_SSL_CTX = ssl.create_default_context()


def _http_get_json(url: str, timeout: int = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout, context=_SSL_CTX) as resp:
        return json.loads(resp.read())


def _api_lookup_ndc(value: str) -> str:
    """RxNorm/NLM API fallback for NDC codes."""
    try:
        d = _http_get_json(f"https://rxnav.nlm.nih.gov/REST/ndcstatus.json?ndc={value}")
        name = d.get("ndcStatus", {}).get("conceptName", "")
        if name:
            return name.title()
    except Exception as e:
        logger.warning("NDC code lookup failed for %r: %s", value, e)
    return ""


def _api_lookup_icd(value: str) -> str:
    """NLM Clinical Tables API fallback for ICD-10 codes."""
    try:
        d = _http_get_json(
            f"https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
            f"?sf=code,name&terms={value}&maxList=1"
        )
        if d and len(d) >= 4 and d[3]:
            return d[3][0][1]
    except Exception as e:
        logger.warning("ICD-10 code lookup failed for %r: %s", value, e)
    return ""


def _api_lookup_cpt(value: str) -> str:
    """NLM HCPCS API fallback for CPT/HCPCS codes."""
    try:
        d = _http_get_json(
            f"https://clinicaltables.nlm.nih.gov/api/hcpcs/v3/search"
            f"?sf=code,display&terms={value}&maxList=1"
        )
        if d and len(d) >= 4 and d[3]:
            return d[3][0][1]
    except Exception as e:
        logger.warning("CPT/HCPCS code lookup failed for %r: %s", value, e)
    return ""

#: ddgs currently NOT used in this project, keeping for backward compatibility.
#   TODO: replace with a more professional/compliant search engine for industries, e.g., code lookup MCP servers.
def web_search(query: str, max_results: int = 3) -> str:
    """Kept for backward compatibility. Performs a DuckDuckGo text search."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore[no-redef]
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        parts = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            parts.append(f"{title}\n{body}\n{href}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Search failed: {e}"


def _is_numeric_like(value: Any) -> bool:
    text = str(value).strip()
    if not text:
        return False
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text))


def _should_skip_code_column(column: str, sample_rows: list[dict]) -> bool:
    """Filter out obvious aggregate or metric columns before enrichment."""
    lower = column.lower()
    if any(token in lower for token in _AGGREGATE_COLUMN_TOKENS):
        return True

    values = [row.get(column) for row in sample_rows if row.get(column) is not None]
    if not values:
        return False

    has_code_hint = any(token in lower for token in _CODE_HINT_TOKENS)
    if has_code_hint:
        return False

    # Pure numeric columns without identifier hints are usually metrics, not codes.
    if all(_is_numeric_like(value) for value in values):
        return True

    return False


def detect_code_columns(
    columns: list[str],
    sample_data: list[dict],
    llm: Any,
) -> list[dict]:
    """Use an LLM to identify which columns contain coded identifiers.

    Returns a list like [{"column": "ndc", "code_type": "NDC"}].
    """
    sample_rows = sample_data[:3]
    sample_str = json.dumps(sample_rows, indent=2, default=str)
    if len(sample_str) > 2000:
        sample_str = sample_str[:2000] + "\n..."

    prompt = (
        "You are a data analyst. Given these column names and sample rows from a "
        "SQL query result, identify which columns contain **coded identifiers** that "
        "would benefit from a human-readable description lookup.\n\n"
        "Examples of coded identifiers: NDC (drug codes), ICD-10/ICD-9 (diagnosis), "
        "CPT/HCPCS (procedures), NAICS/SIC (industry), FIPS (geography), "
        "ticker symbols (stocks), currency codes, zip codes with names, etc.\n\n"
        "Do NOT flag columns that are already human-readable (names, descriptions, "
        "dates, counts, amounts) or generic auto-increment IDs.\n"
        "Do NOT flag aggregate metric columns such as *_count, *_cost, *_amount, "
        "coverage_days, total_rows, procedure_units, averages, or ranks.\n\n"
        f"Columns: {columns}\n\n"
        f"Sample rows:\n{sample_str}\n\n"
        "Return ONLY a JSON array. Each element: "
        '{{"column": "<column_name>", "code_type": "<type>"}}. '
        "If no columns are coded identifiers, return [].\n"
        "Output ONLY the JSON array, nothing else."
    )

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            detected = json.loads(match.group())
            return [
                item for item in detected
                if item.get("column") in columns
                and not _should_skip_code_column(item["column"], sample_rows)
            ]
        return []
    except Exception as e:
        print(f"⚠ detect_code_columns failed: {e}")
        return []


def _sanitize_for_markdown_table(text: str) -> str:
    """Escape characters that break markdown table cells."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("|", "\\|")
    return text.strip()


def _lookup_codes_via_llm(
    code_type: str,
    values: list[str],
    llm: Any,
) -> dict[str, str]:
    """Single batch LLM call to look up descriptions for all unique code values.

    Returns a dict mapping each code value to its description string.
    Unknown codes are marked "(unknown by LLM)".
    """
    if not values:
        return {}

    values_json = json.dumps(values)
    prompt = (
        f"You are a knowledgeable data analyst and domain expert.\n\n"
        f"Look up the human-readable description for each of the following "
        f"**{code_type}** code values.\n\n"
        f"Rules:\n"
        f"- Provide a concise description (under 80 characters) for each code you know "
        f"with HIGH confidence (e.g., drug name + strength + form for NDC; diagnosis "
        f"name for ICD-10; procedure name for CPT; company name for ticker; etc.).\n"
        f"- If you are NOT sure, or the code is not in your training data, "
        f'mark it exactly as: (unknown by LLM)\n'
        f"- NEVER guess or hallucinate. Accuracy is critical — an honest "
        f'"(unknown by LLM)" is far better than a wrong description.\n'
        f"- Do not include the code itself in the description.\n\n"
        f"Code values to look up ({code_type}):\n{values_json}\n\n"
        f"Return ONLY a JSON object mapping each code to its description:\n"
        f'{{"code1": "description or (unknown by LLM)", "code2": "...", ...}}\n'
        f"Output ONLY the JSON object, nothing else."
    )

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"⚠ LLM code lookup failed: {e}")
    return {v: "(unknown by LLM)" for v in values}


def enrich_codes(
    columns: list[str],
    data: list[dict],
    llm: Any,
    writer: Optional[Any] = None,
    section_title: Optional[str] = None,
    wrap_in_details: bool = True,
) -> Optional[str]:
    """Detect coded columns then look up descriptions for unique code values.

    Returns a compact "Code Reference" markdown section mapping unique codes
    to their API-verified descriptions, or None if nothing to enrich.
    """
    if not columns or not data:
        return None

    code_cols = detect_code_columns(columns, data, llm)
    if not code_cols:
        print("ℹ No coded columns detected — skipping enrichment")
        return None

    col_names = [c["column"] for c in code_cols if c["column"] in columns]
    if not col_names:
        return None

    code_type_map = {c["column"]: c["code_type"] for c in code_cols}
    print(f"🔍 Detected coded columns: {col_names}")

    sections: list[str] = []
    for col in col_names:
        unique_vals = list(dict.fromkeys(
            str(row.get(col, "")) for row in data if row.get(col) is not None
        ))[:_MAX_CODES_TO_LOOKUP]

        code_type = code_type_map.get(col, "code")
        print(f"🤖 LLM lookup: {len(unique_vals)} {code_type} values in column '{col}'")

        raw_lookups = _lookup_codes_via_llm(code_type, unique_vals, llm)

        ct_upper = code_type.upper()
        api_fn = None
        if "NDC" in ct_upper:
            api_fn = _api_lookup_ndc
        elif "ICD" in ct_upper:
            api_fn = _api_lookup_icd
        elif "CPT" in ct_upper or "HCPCS" in ct_upper:
            api_fn = _api_lookup_cpt

        if api_fn:
            unknown_vals = [v for v, d in raw_lookups.items() if "(unknown by LLM)" in d]
            if unknown_vals:
                print(f"  🔍 API fallback for {len(unknown_vals)} unknown {code_type} codes")
                for val in unknown_vals:
                    api_desc = api_fn(val)
                    if api_desc:
                        raw_lookups[val] = api_desc

        rows = []
        for code_val, desc in raw_lookups.items():
            safe_code = _sanitize_for_markdown_table(str(code_val))
            safe_desc = _sanitize_for_markdown_table(desc)
            rows.append(f"| {safe_code} | {safe_desc} |")

        if rows:
            section = f"\n**{code_type} Codes** (`{col}`)\n\n| Code | Description |\n|---|---|\n"
            section += "\n".join(rows)
            sections.append(section)

    if not sections:
        return None

    result = ""
    if section_title:
        result += f"\n### {section_title}\n\n"
    result += "\n".join(sections)

    if wrap_in_details:
        result = (
            "\n\n<details name=\"sql-accordion\"><summary>Code Reference</summary>\n\n<div class=\"accordion-content\">\n\n"
            + result
            + "\n\n</div>\n</details>\n"
        )
    print(f"✓ Code reference built ({len(sections)} code type sections)")
    return result
