"""Unit tests for the shared join-contract state artifact."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_server.multi_agent.utils.join_contract import (
    build_join_contract_from_execution,
    build_join_contract_from_genie_results,
    build_join_contract_from_plan,
    enrich_question_with_join_contract,
    format_join_contract_block,
    join_contract_summary,
    merge_join_contracts,
)


def test_build_from_plan_seeds_entities_and_edges():
    plan = {
        "original_query": "top drugs in 2024 and their diagnoses",
        "requires_join": True,
        "dependency_edges": [{"from": "space_a", "to": "space_b"}],
        "genie_route_plan": {
            "space_a": {"question": "Top drugs in 2024", "depends_on": []},
            "space_b": {
                "question": "Diagnoses for those drugs",
                "depends_on": ["space_a"],
            },
        },
        "vector_search_relevant_spaces_info": [
            {"space_id": "space_a", "space_title": "Pharmacy"},
            {"space_id": "space_b", "space_title": "Diagnoses"},
        ],
    }
    contract = build_join_contract_from_plan(plan)
    assert contract["source"] == "planning"
    assert "Pharmacy" in contract["entities"]
    assert contract["dependency_edges"] == [{"from": "space_a", "to": "space_b"}]
    assert "2024" in (contract["time_window"].get("raw") or [])


def test_build_from_genie_results_captures_sql_keys_and_cids():
    results = {
        "space_a": {
            "success": True,
            "sql": "SELECT drug_code FROM drugs WHERE year = 2024 AND drug_code = 'D001'",
            "answer": "Top drug is 'D001'",
            "conversation_id": "conv-a",
        },
        "_genie_dag": {"dependencies": {"space_b": ["space_a"]}},
    }
    contract = build_join_contract_from_genie_results(
        results,
        relevant_spaces=[{"space_id": "space_a", "space_title": "Pharmacy"}],
    )
    assert contract["sql_by_space"]["space_a"].startswith("SELECT")
    assert contract["conversation_ids"]["space_a"] == "conv-a"
    assert "D001" in contract["keys"]
    assert contract["dependency_edges"] == [{"from": "space_a", "to": "space_b"}]


def test_build_from_execution_and_merge():
    execution_results = [
        {
            "success": True,
            "query_label": "Top drugs",
            "columns": ["drug_code", "total_cost"],
            "sql": "SELECT drug_code, total_cost FROM drugs WHERE year >= 2023",
            "result": [
                {"drug_code": "D001", "total_cost": 10},
                {"drug_code": "D002", "total_cost": 8},
            ],
        }
    ]
    exec_contract = build_join_contract_from_execution(execution_results)
    assert "D001" in exec_contract["keys"]
    assert "total_cost" in exec_contract["metrics"]
    assert "drug_code" in exec_contract["key_columns"]

    plan_contract = build_join_contract_from_plan(
        {
            "dependency_edges": [{"from": "a", "to": "b"}],
            "vector_search_relevant_spaces_info": [
                {"space_id": "a", "space_title": "Pharmacy"}
            ],
        }
    )
    merged = merge_join_contracts(plan_contract, exec_contract)
    assert merged["source"] == "merged"
    assert "Pharmacy" in merged["entities"]
    assert "D001" in merged["keys"]
    assert merged["dependency_edges"] == [{"from": "a", "to": "b"}]


def test_format_and_enrich_question():
    contract = merge_join_contracts(
        build_join_contract_from_plan(
            {
                "vector_search_relevant_spaces_info": [
                    {"space_id": "a", "space_title": "Pharmacy"}
                ],
                "dependency_edges": [{"from": "a", "to": "b"}],
            }
        ),
        {
            "version": 1,
            "entities": [],
            "keys": ["D001"],
            "key_columns": ["drug_code"],
            "time_window": {"start": "2024", "end": "2024", "raw": ["2024"]},
            "metrics": ["total_cost"],
            "sql_by_space": {"a": "SELECT 1"},
            "conversation_ids": {"a": "c1"},
            "dependency_edges": [],
            "literals_by_column": {"drug_code": ["D001"]},
            "source": "genie",
            "notes": [],
        },
    )
    block = format_join_contract_block(contract)
    assert "JOIN CONTRACT" in block
    assert "D001" in block
    assert "total_cost" in block

    enriched = enrich_question_with_join_contract("Get diagnoses", contract)
    assert enriched.startswith("Get diagnoses")
    assert "JOIN CONTRACT" in enriched
    assert enrich_question_with_join_contract(enriched, contract) == enriched

    summary = join_contract_summary(contract)
    assert summary["present"] is True
    assert summary["key_count"] >= 1
