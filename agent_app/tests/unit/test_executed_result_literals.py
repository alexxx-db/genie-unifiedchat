"""Unit tests for executed warehouse result → Genie literal injection."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_server.multi_agent.utils.executed_result_literals import (
    build_executed_literal_package,
    enrich_question_with_executed_literals,
    extract_literals_from_rows,
    format_executed_literals_block,
)


def test_extract_literals_from_rows_prefers_code_columns():
    columns = ["drug_code", "total_cost", "drug_name"]
    rows = [
        {"drug_code": "D001", "total_cost": 12.5, "drug_name": "Aspirin"},
        {"drug_code": "D002", "total_cost": 9.0, "drug_name": "Ibuprofen"},
        {"drug_code": "D001", "total_cost": 3.0, "drug_name": "Aspirin"},
    ]
    extracted = extract_literals_from_rows(columns, rows)
    assert extracted["by_column"]["drug_code"] == ["D001", "D002"]
    assert "Aspirin" in extracted["by_column"]["drug_name"]
    # Float metrics are skipped as cell literals.
    assert "total_cost" not in extracted["by_column"]
    assert "D001" in extracted["literals"]


def test_build_package_and_format_block():
    results = [
        {
            "success": True,
            "query_number": 1,
            "query_label": "Top drugs",
            "row_count": 2,
            "columns": ["drug_code", "drug_name"],
            "sql": "SELECT drug_code, drug_name FROM drugs LIMIT 2",
            "result": [
                {"drug_code": "D001", "drug_name": "Aspirin"},
                {"drug_code": "D002", "drug_name": "Ibuprofen"},
            ],
        }
    ]
    package = build_executed_literal_package(results)
    assert package["has_literals"] is True
    assert package["literals"][:2] == ["D001", "D002"]

    block = format_executed_literals_block(package)
    assert "EXECUTED RESULT LITERALS" in block
    assert "drug_code: D001, D002" in block
    assert "IN ('D001', 'D002')" in block
    assert "Top drugs" in block


def test_enrich_question_with_executed_literals():
    package = build_executed_literal_package(
        [
            {
                "success": True,
                "columns": ["member_id"],
                "result": [{"member_id": "M1"}, {"member_id": "M2"}],
                "row_count": 2,
            }
        ]
    )
    enriched = enrich_question_with_executed_literals(
        "Get claims for those members",
        package,
    )
    assert enriched.startswith("Get claims for those members")
    assert "EXECUTED RESULT LITERALS" in enriched
    assert "M1" in enriched
    # Idempotent if already enriched
    assert enrich_question_with_executed_literals(enriched, package) == enriched


def test_skips_failed_and_skipped_results():
    package = build_executed_literal_package(
        [
            {"status": "failed", "columns": ["id"], "result": [{"id": "X"}]},
            {"status": "skipped", "columns": ["id"], "result": [{"id": "Y"}]},
            {
                "success": True,
                "columns": ["id"],
                "result": [{"id": "Z"}],
            },
        ]
    )
    assert package["literals"] == ["Z"]
    assert package["query_count"] == 1


def test_empty_package_formats_as_empty_string():
    assert format_executed_literals_block(None) == ""
    assert format_executed_literals_block({"has_literals": False}) == ""
    assert enrich_question_with_executed_literals("hello", None) == "hello"
