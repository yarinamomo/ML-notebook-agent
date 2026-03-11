from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "data_analysis" / "compare_fixed_notebooks.py"
    spec = importlib.util.spec_from_file_location("compare_fixed_notebooks", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare_fixed_notebooks = load_module()


def write_notebook(path: Path, code_cells: list[str]) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": ["# Title"],
            },
            *[
                {
                    "cell_type": "code",
                    "metadata": {"language": "python"},
                    "execution_count": i + 1,
                    "source": code.splitlines(),
                }
                for i, code in enumerate(code_cells)
            ],
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook), encoding="utf-8")


def write_summary(path: Path, status: str, original_notebook: Sequence[object], code_changes: list[dict]) -> None:
    payload = {
        "metadata": {"status": status},
        "original_notebook": original_notebook,
        "code_changes": code_changes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_extract_original_code_cells_supports_legacy_and_dict_entries():
    cells = compare_fixed_notebooks.extract_original_code_cells(
        [
            "# --- [CELL 0]: ---\n\nprint('legacy')\n",
            {"cell_type": "markdown", "source": ["ignore"]},
            {"cell_type": "code", "source": ["x = 1", "y = 2"]},
        ]
    )

    assert cells == ["print('legacy')\n", "x = 1\ny = 2"]


def test_classify_changes_uses_requested_categories():
    config = compare_fixed_notebooks.ComparisonConfig()
    reference_changes = {0: "a = 2"}
    fixed_cells = ["a = 2", "b = 1"]

    classification, reason, cell_reasons = compare_fixed_notebooks.classify_changes(
        reference_changes,
        {0: "a = 2"},
        fixed_cells,
        config,
    )
    assert classification == compare_fixed_notebooks.VALID
    assert reason == "all_summary_cells_match_change_set_equal"
    assert cell_reasons == {0: "exact_normalized"}

    classification, reason, cell_reasons = compare_fixed_notebooks.classify_changes(
        {0: "a = 2", 1: "b = 2"},
        {0: "a = 2"},
        fixed_cells,
        config,
    )
    assert classification == compare_fixed_notebooks.VALID_WITH_EXTRA_CHANGES
    assert reason == "all_summary_cells_match_change_set_differs"
    assert cell_reasons == {0: "exact_normalized"}

    classification, reason, cell_reasons = compare_fixed_notebooks.classify_changes(
        reference_changes,
        {0: "a = 3"},
        fixed_cells,
        config,
    )
    assert classification == compare_fixed_notebooks.PLAUSIBLE
    assert reason == "summary_cell_mismatch"
    assert cell_reasons == {0: "mismatch"}


def test_cells_match_handles_simple_renamings():
    config = compare_fixed_notebooks.ComparisonConfig()

    assert compare_fixed_notebooks.cells_match(
        "value = 1\nprint(value)",
        "other = 1\nprint(other)",
        config,
    )
    assert (
        compare_fixed_notebooks.cell_match_reason(
            "value = 1\nprint(value)",
            "other = 1\nprint(other)",
            config,
        )
        == "ast_equivalent_renaming"
    )


def test_cells_match_keeps_non_renaming_behavior_differences():
    config = compare_fixed_notebooks.ComparisonConfig()

    assert not compare_fixed_notebooks.cells_match(
        "x = 1\nprint(x)",
        "x = 2\nprint(x)",
        config,
    )


def test_cells_match_ignores_comments_and_empty_lines_via_ast():
    config = compare_fixed_notebooks.ComparisonConfig()

    assert compare_fixed_notebooks.cells_match(
        "x = 1\n\n# note\nprint(x)",
        "x = 1\nprint(x)",
        config,
    )


def test_compute_changed_cells_can_ignore_extra_reference_cells():
    config = compare_fixed_notebooks.ComparisonConfig()

    changes = compare_fixed_notebooks.compute_changed_cells(
        ["a = 1", "b = 1"],
        ["a = 2", "b = 1", "helper = 0"],
        config,
        limit=2,
    )

    assert changes == {0: "a = 2"}


def test_analyze_results_directory_reports_all_three_categories(tmp_path: Path):
    results_dir = tmp_path / "results"
    benchmark_dir = tmp_path / "JunoBench" / "benchmark"

    valid_original = ["# --- [CELL 0]: ---\n\na = 1\n", "# --- [CELL 1]: ---\n\nb = 1\n"]
    extra_original = ["# --- [CELL 0]: ---\n\nx = 1\n", "# --- [CELL 1]: ---\n\ny = 1\n"]
    plausible_original = ["# --- [CELL 0]: ---\n\nm = 1\n"]

    write_notebook(
        benchmark_dir / "valid_case" / "valid_case_fixed.ipynb",
        ["a = 2\n\n", "b = 1"],
    )
    write_notebook(benchmark_dir / "extra_case" / "extra_case_fixed.ipynb", ["x = 2", "y = 1"])
    write_notebook(benchmark_dir / "plausible_case" / "plausible_case_fixed.ipynb", ["m = 2"])

    write_summary(
        results_dir / "agent" / "model-a" / "run_1" / "valid_case_summary.json",
        "Submitted",
        valid_original,
        [{"cell_index": 0, "step": 1, "code": "a = 2"}],
    )
    write_summary(
        results_dir / "agent" / "model-a" / "run_1" / "extra_case_summary.json",
        "SubmittedWithErrors",
        extra_original,
        [
            {"cell_index": 0, "step": 1, "code": "x = 2"},
            {"cell_index": 1, "step": 2, "code": "y = 2"},
        ],
    )
    write_summary(
        results_dir / "agent" / "model-a" / "run_1" / "plausible_case_summary.json",
        "Submitted",
        plausible_original,
        [{"cell_index": 0, "step": 1, "code": "m = 3"}],
    )
    write_summary(
        results_dir / "agent" / "model-a" / "run_1" / "ignored_case_summary.json",
        "Incomplete",
        plausible_original,
        [{"cell_index": 0, "step": 1, "code": "m = 2"}],
    )

    report = compare_fixed_notebooks.analyze_results_directory(results_dir, benchmark_dir)
    records = {record["instance"]: record for record in report["records"]}

    assert report["statistics"]["summaries_compared"] == 3
    assert report["statistics"]["classification_counts"] == {
        compare_fixed_notebooks.VALID: 1,
        compare_fixed_notebooks.VALID_WITH_EXTRA_CHANGES: 0,
        compare_fixed_notebooks.PLAUSIBLE: 2,
    }
    assert records["valid_case"]["classification"] == compare_fixed_notebooks.VALID
    assert records["valid_case"]["match_reason"] == "all_summary_cells_match_change_set_equal"
    assert records["valid_case"]["summary_cell_match_reasons"] == {"0": "exact_normalized"}
    assert records["valid_case"]["original_code_cell_count"] == 2
    assert records["valid_case"]["reference_code_cell_count"] == 2
    assert records["extra_case"]["classification"] == compare_fixed_notebooks.PLAUSIBLE
    assert records["extra_case"]["comparison"]["extra_summary_changes"] == [1]
    assert records["plausible_case"]["classification"] == compare_fixed_notebooks.PLAUSIBLE
    assert records["plausible_case"]["comparison"]["mismatched_shared_changes"] == [0]


def test_analyze_results_directory_ignores_reference_cells_beyond_original_range(tmp_path: Path):
    results_dir = tmp_path / "results"
    benchmark_dir = tmp_path / "JunoBench" / "benchmark"

    original = ["# --- [CELL 0]: ---\n\na = 1\n", "# --- [CELL 1]: ---\n\nb = 1\n"]

    write_notebook(
        benchmark_dir / "range_case" / "range_case_fixed.ipynb",
        ["a = 2", "b = 1", "helper = 0"],
    )
    write_summary(
        results_dir / "agent" / "model-a" / "run_1" / "range_case_summary.json",
        "Submitted",
        original,
        [{"cell_index": 0, "step": 1, "code": "a = 2"}],
    )

    report = compare_fixed_notebooks.analyze_results_directory(results_dir, benchmark_dir)
    record = report["records"][0]

    assert record["classification"] == compare_fixed_notebooks.VALID
    assert record["match_reason"] == "all_summary_cells_match_change_set_equal"
    assert record["original_code_cell_count"] == 2
    assert record["reference_code_cell_count"] == 3
    assert record["reference_changed_cells"] == [0]


def test_alignment_handles_prepended_fixed_setup_cell(tmp_path: Path):
    results_dir = tmp_path / "results"
    benchmark_dir = tmp_path / "JunoBench" / "benchmark"

    original = [
        "# --- [CELL 0]: ---\n\nimport numpy as np\n",
        "# --- [CELL 1]: ---\n\na = 1\n",
        "# --- [CELL 2]: ---\n\nb = 1\n",
    ]

    write_notebook(
        benchmark_dir / "setup_shift_case" / "setup_shift_case_fixed.ipynb",
        [
            "!pip install qiskit",
            "import numpy as np",
            "a = 2",
            "b = 1",
        ],
    )
    write_summary(
        results_dir / "agent" / "model-a" / "run_1" / "setup_shift_case_summary.json",
        "Submitted",
        original,
        [{"cell_index": 1, "step": 1, "code": "a = 2"}],
    )

    report = compare_fixed_notebooks.analyze_results_directory(results_dir, benchmark_dir)
    record = report["records"][0]

    assert record["classification"] == compare_fixed_notebooks.VALID
    assert record["match_reason"] == "all_summary_cells_match_change_set_equal"
    assert record["summary_cell_match_reasons"] == {"1": "exact_normalized"}
    assert record["alignment_applied"] is True