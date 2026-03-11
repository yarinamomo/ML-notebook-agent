"""Aggregate analysis for edit_cell behavior in trajectories_without_run_code."""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


class EditCellAnalyzer:
    """Analyze edit_cell statistics across all *_summary.json files."""

    def __init__(self, root_directory: Path) -> None:
        self.root_directory = root_directory
        self.results: dict[str, Any] = {
            "metadata": {
                "root_directory": str(root_directory.name),
                "files_analyzed": 0,
                "files_with_cell_edits": 0,
                "instances_without_cell_edits": [],
            },
            "summary": {
                "avg_cell_edits": 0.0,
                "max_cell_edits": 0,
                "max_cell_edit_instances": [],
                "avg_unique_cell_edits": 0.0,
                "avg_cell_edits_with_added_prints": 0.0,
                "avg_cell_edits_with_added_try_excepts": 0.0,
                "by_run": {},
                "examples_cell_edits_with_added_prints": [],
                "examples_cell_edits_with_added_try_excepts": [],
            },
        }

    def find_summary_files(self) -> list[Path]:
        return sorted(self.root_directory.glob("**/run_*/*_summary.json"))

    @staticmethod
    def parse_cell_code(cell_blob: str) -> str:
        """Extract executable code from original_notebook cell blob."""
        if not isinstance(cell_blob, str):
            return ""
        parts = cell_blob.split("\n\n", 1)
        return parts[1] if len(parts) == 2 else cell_blob

    @staticmethod
    def normalize_print_source(source: str) -> str:
        return re.sub(r"\s+", "", source)

    def extract_print_calls(self, code: str) -> Counter[str]:
        """Return multiset of print(...) calls found in code."""
        if not code:
            return Counter()

        try:
            tree = ast.parse(code)
            calls: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                    src = ast.get_source_segment(code, node)
                    calls.append(self.normalize_print_source(src or "print()"))
            return Counter(calls)
        except SyntaxError:
            lines = []
            for line in code.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and re.search(r"\bprint\s*\(", stripped):
                    lines.append(self.normalize_print_source(stripped))
            return Counter(lines)

    def short_instance_name(self, relative_path: Path) -> str:
        """Return path in requested style: run_x/filename.json when possible."""
        parts = relative_path.parts
        run_idx = next((i for i, part in enumerate(parts) if part.startswith("run_")), None)
        if run_idx is not None and run_idx + 1 < len(parts):
            return f"{parts[run_idx]}/{parts[-1]}"
        return str(relative_path).replace("\\", "/")

    @staticmethod
    def run_name_from_path(relative_path: Path) -> str:
        for part in relative_path.parts:
            if part.startswith("run_"):
                return part
        return "unknown"

    def extract_try_except_blocks(self, code: str) -> list[str]:
        """Extract all try-except block source code from code."""
        if not code:
            return []

        blocks: list[str] = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    src = ast.get_source_segment(code, node)
                    if src:
                        blocks.append(src.strip())
            return blocks
        except SyntaxError:
            return []

    def extract_print_call_counter_and_raw(self, code: str) -> tuple[Counter[str], dict[str, list[str]]]:
        """Return print call multiset and raw-source mapping keyed by normalized print source."""
        if not code:
            return Counter(), {}

        counter: Counter[str] = Counter()
        raw_map: dict[str, list[str]] = {}

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                    src = ast.get_source_segment(code, node) or "print()"
                    key = self.normalize_print_source(src)
                    counter[key] += 1
                    raw_map.setdefault(key, []).append(src.strip())
            return counter, raw_map
        except SyntaxError:
            for line in code.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and re.search(r"\bprint\s*\(", stripped):
                    key = self.normalize_print_source(stripped)
                    counter[key] += 1
                    raw_map.setdefault(key, []).append(stripped)
            return counter, raw_map

    def analyze(self) -> dict[str, Any]:
        summary_files = self.find_summary_files()
        total_files = len(summary_files)

        self.results["metadata"]["files_analyzed"] = total_files

        edits_per_file: list[int] = []
        unique_cell_edits_per_file: list[int] = []
        edits_with_added_prints_per_file: list[int] = []
        edits_with_added_try_excepts_per_file: list[int] = []
        edits_by_instance: list[tuple[str, int]] = []
        run_accumulator: dict[str, dict[str, list[int]]] = {}
        examples_with_added_prints: list[dict[str, Any]] = []
        examples_with_added_try_excepts: list[dict[str, Any]] = []

        for file_path in summary_files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.results["metadata"]["instances_without_cell_edits"].append(
                    {"file": str(file_path.relative_to(self.root_directory)).replace("\\", "/"), "error": str(exc)}
                )
                continue

            code_changes = data.get("code_changes", []) or []
            original_notebook = data.get("original_notebook", []) or []
            rel_path = file_path.relative_to(self.root_directory)
            run_name = self.run_name_from_path(rel_path)
            instance_name = self.short_instance_name(rel_path)

            file_edits = len(code_changes)
            file_unique_cells = len(
                {
                    change.get("cell_index")
                    for change in code_changes
                    if isinstance(change.get("cell_index"), int)
                }
            )

            file_added_print_edit_count = 0
            file_added_try_except_edit_count = 0
            for change in code_changes:
                cell_index = change.get("cell_index")
                changed_code = change.get("code", "")
                step = change.get("step")

                if not isinstance(cell_index, int) or not isinstance(changed_code, str):
                    continue

                original_code = ""
                if 0 <= cell_index < len(original_notebook):
                    original_code = self.parse_cell_code(original_notebook[cell_index])

                original_prints, _ = self.extract_print_call_counter_and_raw(original_code)
                changed_prints, changed_raw_map = self.extract_print_call_counter_and_raw(changed_code)

                newly_added_prints = changed_prints - original_prints
                if sum(newly_added_prints.values()) > 0:
                    file_added_print_edit_count += 1
                    if len(examples_with_added_prints) < 5:
                        added_prints_readable: list[str] = []
                        for key, count in newly_added_prints.items():
                            if count <= 0:
                                continue
                            raw_values = changed_raw_map.get(key, [key])
                            value = raw_values[0]
                            added_prints_readable.extend([value] * count)

                        examples_with_added_prints.append(
                            {
                                "instance_name": instance_name,
                                "cell_index": cell_index,
                                "step": step,
                                "added_prints": added_prints_readable,
                            }
                        )

                original_try_excepts = set(self.extract_try_except_blocks(original_code))
                changed_try_excepts = set(self.extract_try_except_blocks(changed_code))
                newly_added_try_excepts = changed_try_excepts - original_try_excepts

                if len(newly_added_try_excepts) > 0:
                    file_added_try_except_edit_count += 1
                    if len(examples_with_added_try_excepts) < 5:
                        examples_with_added_try_excepts.append(
                            {
                                "instance_name": instance_name,
                                "cell_index": cell_index,
                                "step": step,
                                "added_try_excepts": list(newly_added_try_excepts)[:2],
                            }
                        )

            edits_per_file.append(file_edits)
            edits_by_instance.append((instance_name, file_edits))
            unique_cell_edits_per_file.append(file_unique_cells)
            edits_with_added_prints_per_file.append(file_added_print_edit_count)
            edits_with_added_try_excepts_per_file.append(file_added_try_except_edit_count)

            run_accumulator.setdefault(
                run_name,
                {
                    "edits": [],
                    "unique_cells": [],
                    "added_print_edits": [],
                    "added_try_except_edits": [],
                },
            )
            run_accumulator[run_name]["edits"].append(file_edits)
            run_accumulator[run_name]["unique_cells"].append(file_unique_cells)
            run_accumulator[run_name]["added_print_edits"].append(file_added_print_edit_count)
            run_accumulator[run_name]["added_try_except_edits"].append(file_added_try_except_edit_count)

            if file_edits == 0:
                self.results["metadata"]["instances_without_cell_edits"].append(instance_name)

        files_without = self.results["metadata"]["instances_without_cell_edits"]
        self.results["metadata"]["instances_without_cell_edits"] = sorted(files_without)
        self.results["metadata"]["files_with_cell_edits"] = (
            self.results["metadata"]["files_analyzed"] - len(files_without)
        )

        divisor = len(edits_per_file) if edits_per_file else 1
        self.results["summary"]["avg_cell_edits"] = round(sum(edits_per_file) / divisor, 4)
        max_cell_edits = max(edits_per_file) if edits_per_file else 0
        self.results["summary"]["max_cell_edits"] = max_cell_edits
        self.results["summary"]["max_cell_edit_instances"] = (
            sorted(instance for instance, edits in edits_by_instance if edits == max_cell_edits)
            if max_cell_edits > 1
            else []
        )
        self.results["summary"]["avg_unique_cell_edits"] = round(sum(unique_cell_edits_per_file) / divisor, 4)
        self.results["summary"]["avg_cell_edits_with_added_prints"] = round(
            sum(edits_with_added_prints_per_file) / divisor,
            4,
        )
        self.results["summary"]["avg_cell_edits_with_added_try_excepts"] = round(
            sum(edits_with_added_try_excepts_per_file) / divisor,
            4,
        )
        self.results["summary"]["examples_cell_edits_with_added_prints"] = examples_with_added_prints
        self.results["summary"]["examples_cell_edits_with_added_try_excepts"] = examples_with_added_try_excepts

        by_run: dict[str, dict[str, float]] = {}
        for run_name, values in sorted(run_accumulator.items()):
            run_divisor = len(values["edits"]) if values["edits"] else 1
            by_run[run_name] = {
                "avg_cell_edits": round(sum(values["edits"]) / run_divisor, 4),
                "avg_unique_cell_edits": round(sum(values["unique_cells"]) / run_divisor, 4),
                "avg_cell_edits_with_added_prints": round(sum(values["added_print_edits"]) / run_divisor, 4),
                "avg_cell_edits_with_added_try_excepts": round(sum(values["added_try_except_edits"]) / run_divisor, 4),
            }
        self.results["summary"]["by_run"] = by_run

        return self.results


def main(target_dir: Path) -> None:
    if not target_dir.exists():
        print(f"Error: Directory not found: {target_dir}")
        return

    analyzer = EditCellAnalyzer(target_dir)
    results = analyzer.analyze()

    # make dir if not exists
    output_dir = target_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "edit_cell_analysis_results.json"
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Analysis complete")
    print(f"files_analyzed: {results['metadata']['files_analyzed']}")
    print(f"files_without_cell_edits: {len(results['metadata']['instances_without_cell_edits'])}")
    print(f"avg_cell_edits: {results['summary']['avg_cell_edits']}")
    print(f"max_cell_edits: {results['summary']['max_cell_edits']}")
    if results["summary"]["max_cell_edit_instances"]:
        print("max_cell_edit_instances:")
        for instance in results["summary"]["max_cell_edit_instances"]:
            print(f"  - {instance}")
    print(f"avg_unique_cell_edits: {results['summary']['avg_unique_cell_edits']}")
    print(f"avg_cell_edits_with_added_prints: {results['summary']['avg_cell_edits_with_added_prints']}")
    print(f"avg_cell_edits_with_added_try_excepts: {results['summary']['avg_cell_edits_with_added_try_excepts']}")
    print(f"Saved: {output_file}")
