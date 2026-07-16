"""Unit tests for Genie route DAG normalize / waves / structured inject."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_server.multi_agent.utils.genie_route_dag import (
    apply_dependency_edges,
    build_context_package,
    compute_execution_waves,
    dependency_edges_from_plan,
    enrich_question_with_context,
    ensure_linear_dependencies,
    extract_conversation_ids,
    finalize_planner_genie_plan,
    format_inject_block,
    legacy_question_map,
    merge_conversation_ids,
    normalize_dependency_edges,
    normalize_genie_route_plan,
    query_suggests_staged_dependencies,
    questions_for_wave,
    resolve_genie_execution_mode,
    resolve_space_conversation_id,
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


def test_query_suggests_staged_dependencies():
    assert query_suggests_staged_dependencies(
        "top 10 drugs and their diagnoses"
    )
    assert query_suggests_staged_dependencies(
        "find ndc codes then look up descriptions"
    )
    assert query_suggests_staged_dependencies(
        query="members",
        sub_questions=["highest cost members", "then their claims"],
    )
    assert not query_suggests_staged_dependencies(
        "how many active members? what is total lexapro cost?"
    )


def test_dependency_edges_helpers():
    normalized = normalize_genie_route_plan(
        {
            "a": {"question": "q1", "depends_on": []},
            "b": {"question": "q2", "depends_on": ["a"]},
        }
    )
    edges = dependency_edges_from_plan(normalized)
    assert edges == [{"from": "a", "to": "b"}]

    cleaned = normalize_dependency_edges(
        [{"from": "a", "to": "b"}, {"from": "a", "to": "a"}, {"from": "x", "to": "b"}],
        known_space_ids={"a", "b"},
    )
    assert cleaned == [{"from": "a", "to": "b"}]

    applied = apply_dependency_edges(
        normalize_genie_route_plan({"a": "q1", "b": "q2"}),
        [{"from": "a", "to": "b"}],
    )
    assert applied["b"]["depends_on"] == ["a"]


def test_ensure_linear_dependencies_respects_order():
    normalized = normalize_genie_route_plan(
        {"b": "second", "a": "first", "c": "third"}
    )
    chained = ensure_linear_dependencies(normalized, ["a", "b", "c"])
    assert chained["a"]["depends_on"] == []
    assert chained["b"]["depends_on"] == ["a"]
    assert chained["c"]["depends_on"] == ["b"]


def test_finalize_planner_promotes_staged_flat_plan_to_dag():
    plan = {
        "original_query": "top 10 drugs and their diagnoses",
        "join_strategy": "genie_route",
        "relevant_space_ids": ["space_drugs", "space_dx"],
        "sub_questions": ["top 10 drugs", "diagnoses for those drugs"],
        "genie_execution_mode": "parallel",
        "genie_route_plan": {
            "space_drugs": "Top 10 drugs. Please limit to top 10 rows",
            "space_dx": "Diagnoses. Please limit to top 10 rows",
        },
        "dependency_edges": [],
    }
    finalized = finalize_planner_genie_plan(plan)
    assert finalized["genie_execution_mode"] == "dag"
    assert finalized["genie_route_plan"]["space_dx"]["depends_on"] == ["space_drugs"]
    assert finalized["dependency_edges"] == [
        {"from": "space_drugs", "to": "space_dx"}
    ]


def test_finalize_planner_merges_explicit_edges():
    plan = {
        "original_query": "member demographics and pharmacy costs",
        "join_strategy": "genie_route",
        "relevant_space_ids": ["s1", "s2"],
        "genie_execution_mode": "dag",
        "genie_route_plan": {
            "s1": {"question": "demographics", "depends_on": []},
            "s2": {"question": "costs", "depends_on": []},
        },
        "dependency_edges": [{"from": "s1", "to": "s2"}],
    }
    finalized = finalize_planner_genie_plan(plan)
    assert finalized["genie_route_plan"]["s2"]["depends_on"] == ["s1"]
    assert finalized["dependency_edges"] == [{"from": "s1", "to": "s2"}]
    assert finalized["genie_execution_mode"] == "dag"


def test_finalize_planner_clears_genie_fields_for_table_route():
    plan = {
        "join_strategy": "table_route",
        "genie_execution_mode": "dag",
        "genie_route_plan": {"a": "q"},
        "dependency_edges": [{"from": "a", "to": "b"}],
    }
    finalized = finalize_planner_genie_plan(plan)
    assert finalized["genie_route_plan"] is None
    assert finalized["genie_execution_mode"] is None
    assert finalized["dependency_edges"] == []


def test_finalize_planner_keeps_independent_parallel():
    plan = {
        "original_query": "How many active members? What is total lexapro cost?",
        "join_strategy": "genie_route",
        "relevant_space_ids": ["members", "pharmacy"],
        "sub_questions": [
            "How many active members?",
            "What is total lexapro cost?",
        ],
        "genie_execution_mode": "parallel",
        "genie_route_plan": {
            "members": "Active member count. Please limit to top 10 rows",
            "pharmacy": "Lexapro total cost. Please limit to top 10 rows",
        },
    }
    finalized = finalize_planner_genie_plan(plan)
    assert finalized["genie_execution_mode"] == "parallel"
    assert finalized["dependency_edges"] == []
    assert finalized["genie_route_plan"]["members"]["depends_on"] == []
    assert finalized["genie_route_plan"]["pharmacy"]["depends_on"] == []


def test_extract_and_merge_conversation_ids():
    results = {
        "space_a": {"conversation_id": "conv-a", "sql": "SELECT 1", "success": True},
        "space_b": {"conversation_id": "", "error": "failed", "success": False},
        "_genie_dag": {"mode": "dag"},
    }
    extracted = extract_conversation_ids(results)
    assert extracted == {"space_a": "conv-a"}

    merged = merge_conversation_ids(
        {"space_a": "old-a", "space_c": "conv-c"},
        extracted,
        {"space_a": "conv-a-new"},
    )
    assert merged == {
        "space_a": "conv-a-new",
        "space_c": "conv-c",
    }


def test_resolve_space_conversation_id_prefers_explicit():
    cached = {"space_a": "cached-a"}
    assert (
        resolve_space_conversation_id(
            "space_a",
            explicit="explicit-a",
            cached=cached,
        )
        == "explicit-a"
    )
    assert (
        resolve_space_conversation_id(
            "space_a",
            explicit=None,
            cached=cached,
        )
        == "cached-a"
    )
    assert (
        resolve_space_conversation_id(
            "space_b",
            explicit="  ",
            cached=cached,
        )
        is None
    )
