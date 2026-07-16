"""Unit tests for Genie route DAG normalize / waves / structured inject."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_server.multi_agent.utils.genie_route_dag import (
    build_context_package,
    compute_execution_waves,
    enrich_question_with_context,
    format_inject_block,
    legacy_question_map,
    normalize_genie_route_plan,
    questions_for_wave,
    resolve_genie_execution_mode,
)


def test_normalize_legacy_string_plan():
    plan = {
        "space_a": "Get top drugs. Please limit to top 10 rows",
        "space_b": "Get diagnoses. Please limit to top 10 rows",
    }
    normalized = normalize_genie_route_plan(plan)
    assert set(normalized) == {"space_a", "space_b"}
    assert normalized["space_a"]["question"].startswith("Get top drugs")
    assert normalized["space_a"]["depends_on"] == []
    assert "ids" in normalized["space_a"]["inject"]


def test_normalize_structured_plan_and_self_dep_removed():
    plan = {
        "space_a": {
            "question": "Top drugs",
            "depends_on": ["space_a", "missing_space"],
            "inject": ["ids", "filters"],
        },
        "space_b": {
            "partial_question": "Diagnoses for those drugs",
            "dependencies": ["space_a"],
            "inject": ["ids", "bogus_field"],
        },
    }
    normalized = normalize_genie_route_plan(plan)
    assert normalized["space_a"]["depends_on"] == ["missing_space"]
    assert normalized["space_b"]["question"] == "Diagnoses for those drugs"
    assert normalized["space_b"]["depends_on"] == ["space_a"]
    assert normalized["space_b"]["inject"] == ["ids"]


def test_resolve_mode_infers_dag_from_depends_on():
    normalized = normalize_genie_route_plan(
        {
            "a": {"question": "q1", "depends_on": []},
            "b": {"question": "q2", "depends_on": ["a"]},
        }
    )
    assert resolve_genie_execution_mode(None, normalized) == "dag"
    assert resolve_genie_execution_mode("parallel", normalized) == "dag"
    assert resolve_genie_execution_mode("dag", normalized) == "dag"


def test_resolve_mode_parallel_when_no_deps():
    normalized = normalize_genie_route_plan(
        {"a": "q1", "b": "q2"}
    )
    assert resolve_genie_execution_mode(None, normalized) == "parallel"
    assert resolve_genie_execution_mode("parallel", normalized) == "parallel"


def test_compute_execution_waves_two_levels():
    normalized = normalize_genie_route_plan(
        {
            "a": {"question": "q1", "depends_on": []},
            "b": {"question": "q2", "depends_on": []},
            "c": {"question": "q3", "depends_on": ["a", "b"]},
        }
    )
    waves = compute_execution_waves(normalized)
    assert waves[0] == ["a", "b"]
    assert waves[1] == ["c"]


def test_compute_execution_waves_ignores_unknown_deps():
    normalized = normalize_genie_route_plan(
        {
            "a": {"question": "q1", "depends_on": ["ghost"]},
            "b": {"question": "q2", "depends_on": ["a"]},
        }
    )
    waves = compute_execution_waves(normalized)
    # "ghost" is ignored, so a is ready immediately
    assert waves[0] == ["a"]
    assert waves[1] == ["b"]


def test_build_context_package_extracts_ids_and_filters():
    result = {
        "success": True,
        "answer": "Top drugs include 'DRUG_A' and DRUG99X",
        "sql": "SELECT * FROM drugs WHERE drug_code = 'DRUG_A' AND year >= 2020",
        "question": "top drugs",
    }
    pkg = build_context_package("space_a", result)
    assert pkg["success"] is True
    assert "DRUG_A" in pkg["ids"]
    assert any("drug_code" in f.lower() for f in pkg["filters"])
    assert "DRUG_A" in pkg["sql_preview"]


def test_enrich_question_injects_upstream_context():
    normalized = normalize_genie_route_plan(
        {
            "a": {"question": "Top drugs", "depends_on": []},
            "b": {
                "question": "Diagnoses for those drugs",
                "depends_on": ["a"],
                "inject": ["ids", "answer_summary"],
            },
        }
    )
    packages = {
        "a": build_context_package(
            "a",
            {
                "success": True,
                "answer": "Top drug is 'ASPIRIN'",
                "sql": "SELECT code FROM drugs WHERE code = 'ASPIRIN'",
            },
        )
    }
    enriched = enrich_question_with_context(normalized["b"], packages)
    assert enriched.startswith("Diagnoses for those drugs")
    assert "CONTEXT FROM UPSTREAM GENIE SPACES" in enriched
    assert "ASPIRIN" in enriched
    assert "Answer summary:" in enriched


def test_questions_for_wave_only_enriches_dependents():
    normalized = normalize_genie_route_plan(
        {
            "a": {"question": "Root question", "depends_on": []},
            "b": {
                "question": "Child question",
                "depends_on": ["a"],
                "inject": ["ids"],
            },
        }
    )
    packages = {
        "a": build_context_package(
            "a",
            {"success": True, "answer": "id 'X1'", "sql": "SELECT 1"},
        )
    }
    wave1 = questions_for_wave(["a"], normalized, {})
    assert wave1["a"] == "Root question"

    wave2 = questions_for_wave(["b"], normalized, packages)
    assert "Child question" in wave2["b"]
    assert "X1" in wave2["b"]


def test_format_inject_block_handles_missing_upstream():
    step = {
        "question": "q",
        "depends_on": ["missing"],
        "inject": ["ids"],
    }
    block = format_inject_block(step, {})
    assert "no upstream result available" in block


def test_legacy_question_map():
    normalized = normalize_genie_route_plan(
        {
            "a": {"question": "Q-A", "depends_on": ["b"]},
            "b": "Q-B",
        }
    )
    assert legacy_question_map(normalized) == {"a": "Q-A", "b": "Q-B"}
