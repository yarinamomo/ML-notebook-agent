"""Compare submitted summaries against benchmark fixed notebooks."""

from __future__ import annotations

import ast
from difflib import SequenceMatcher
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from src.utils.nbformat_helper import load_and_parse_notebook
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.utils.nbformat_helper import load_and_parse_notebook


VALID = "Valid"
VALID_WITH_EXTRA_CHANGES = "Valid With Extra Changes"
PLAUSIBLE = "Plausible"
SUBMIT_STATUSES = {"SUBMITTED", "SUBMITTEDWITHERRORS"}
RESULTS_DIR = Path("results")
BENCHMARK_DIR = Path("JunoBench/benchmark")
OUTPUT_PATH = Path("results/data_analysis/fixed_notebook_comparison.json")

SimilarityFunction = Callable[[str, str], float]


@dataclass(slots=True)
class ComparisonConfig:
    ignore_empty_lines: bool = True
    similarity_function: SimilarityFunction | None = None
    similarity_threshold: float = 1.0


def is_submit_status(status: str) -> bool:
    return status.upper() in SUBMIT_STATUSES


def source_to_text(source: Any) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "\n".join(str(part).rstrip("\n") for part in source)
    return ""


def extract_original_code_cells(original_notebook: list[Any]) -> list[str]:
    code_cells: list[str] = []

    for entry in original_notebook:
        if isinstance(entry, str):
            _, separator, code = entry.partition("\n\n")
            code_cells.append(code if separator else entry)
            continue

        if isinstance(entry, dict) and entry.get("cell_type") == "code":
            code_cells.append(source_to_text(entry.get("source", "")))

    return code_cells


def load_notebook_code_cells(notebook_path: Path) -> list[str]:
    notebook = load_and_parse_notebook(notebook_path, "JunoBench_Buggy")
    return [source_to_text(cell.get("source", "")) for cell in notebook.cells]


def normalize_code(code: str, ignore_empty_lines: bool = True) -> str:
    lines = code.splitlines()
    if ignore_empty_lines:
        lines = [line.rstrip() for line in lines if line.strip()]
    return "\n".join(lines).strip()


def ast_dump(code: str, ignore_empty_lines: bool = True) -> str | None:
    normalized_code = normalize_code(code, ignore_empty_lines=ignore_empty_lines)
    if not normalized_code:
        return ""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(normalized_code)
        return ast.dump(tree, annotate_fields=False, include_attributes=False)
    except SyntaxError:
        return None


class RenameNormalizer(ast.NodeTransformer):
    """Canonicalize simple local names so pure renamings can match."""

    def __init__(self) -> None:
        self.scope_stack: list[dict[str, str]] = [dict()]
        self.counter_stack: list[int] = [0]

    def _push_scope(self) -> None:
        self.scope_stack.append(dict())
        self.counter_stack.append(0)

    def _pop_scope(self) -> None:
        self.scope_stack.pop()
        self.counter_stack.pop()

    def _bind_name(self, name: str) -> str:
        scope = self.scope_stack[-1]
        if name in scope:
            return scope[name]

        counter = self.counter_stack[-1] + 1
        self.counter_stack[-1] = counter
        canonical_name = f"v{counter}"
        scope[name] = canonical_name
        return canonical_name

    def _lookup_name(self, name: str) -> str | None:
        for scope in reversed(self.scope_stack):
            if name in scope:
                return scope[name]
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._push_scope()
        node.args = self.visit(node.args)
        node.body = [self.visit(stmt) for stmt in node.body]
        if node.returns:
            node.returns = self.visit(node.returns)
        self._pop_scope()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self._push_scope()
        node.args = self.visit(node.args)
        node.body = [self.visit(stmt) for stmt in node.body]
        if node.returns:
            node.returns = self.visit(node.returns)
        self._pop_scope()
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        self._push_scope()
        node.args = self.visit(node.args)
        node.body = self.visit(node.body)
        self._pop_scope()
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._bind_name(node.arg)
        if node.annotation:
            node.annotation = self.visit(node.annotation)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Store):
            node.id = self._bind_name(node.id)
            return node

        if isinstance(node.ctx, ast.Load):
            mapped = self._lookup_name(node.id)
            if mapped is not None:
                node.id = mapped
        return node


def rename_tolerant_ast_dump(code: str, ignore_empty_lines: bool = True) -> str | None:
    normalized_code = normalize_code(code, ignore_empty_lines=ignore_empty_lines)
    if not normalized_code:
        return ""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(normalized_code)
        canonical = RenameNormalizer().visit(tree)
        ast.fix_missing_locations(canonical)
        return ast.dump(canonical, annotate_fields=False, include_attributes=False)
    except SyntaxError:
        return None


def cell_match_reason(left: str, right: str, config: ComparisonConfig) -> str:
    normalized_left = normalize_code(left, ignore_empty_lines=config.ignore_empty_lines)
    normalized_right = normalize_code(right, ignore_empty_lines=config.ignore_empty_lines)

    if normalized_left == normalized_right:
        return "exact_normalized"

    left_ast = ast_dump(left, ignore_empty_lines=config.ignore_empty_lines)
    right_ast = ast_dump(right, ignore_empty_lines=config.ignore_empty_lines)
    if left_ast is not None and right_ast is not None and left_ast == right_ast:
        return "ast_equivalent"

    left_renamed_ast = rename_tolerant_ast_dump(left, ignore_empty_lines=config.ignore_empty_lines)
    right_renamed_ast = rename_tolerant_ast_dump(right, ignore_empty_lines=config.ignore_empty_lines)
    if (
        left_renamed_ast is not None
        and right_renamed_ast is not None
        and left_renamed_ast == right_renamed_ast
    ):
        return "ast_equivalent_renaming"

    if config.similarity_function is None:
        return "mismatch"

    similarity = config.similarity_function(normalized_left, normalized_right)
    if similarity >= config.similarity_threshold:
        return "similarity_threshold"
    return "mismatch"


def cells_match(left: str, right: str, config: ComparisonConfig) -> bool:
    return cell_match_reason(left, right, config) != "mismatch"


def apply_code_changes(original_cells: list[str], code_changes: list[dict[str, Any]]) -> list[str]:
    final_cells = list(original_cells)

    def sort_key(change: dict[str, Any]) -> tuple[int, int]:
        step = change.get("step")
        cell_index = change.get("cell_index")
        safe_step = step if isinstance(step, int) else 10**9
        safe_index = cell_index if isinstance(cell_index, int) else 10**9
        return safe_step, safe_index

    for change in sorted(code_changes, key=sort_key):
        cell_index = change.get("cell_index")
        code = change.get("code")
        if not isinstance(cell_index, int) or not isinstance(code, str):
            continue

        while len(final_cells) <= cell_index:
            final_cells.append("")
        final_cells[cell_index] = code

    return final_cells


def compute_changed_cells(
    original_cells: list[str],
    updated_cells: list[str],
    config: ComparisonConfig,
    limit: int | None = None,
) -> dict[int, str]:
    changed_cells: dict[int, str] = {}
    max_len = limit if limit is not None else max(len(original_cells), len(updated_cells))

    for cell_index in range(max_len):
        original = original_cells[cell_index] if cell_index < len(original_cells) else ""
        updated = updated_cells[cell_index] if cell_index < len(updated_cells) else ""
        if not cells_match(original, updated, config):
            changed_cells[cell_index] = updated

    return changed_cells


def align_fixed_cells_to_original(
    original_cells: list[str],
    fixed_cells: list[str],
    config: ComparisonConfig,
) -> list[str]:
    """Align fixed notebook cells to original cell indices.

    Uses monotonic anchor matching on unchanged cells to account for inserted
    setup cells (e.g. !pip installs) in fixed notebooks.
    """
    normalized_original = [
        normalize_code(code, ignore_empty_lines=config.ignore_empty_lines)
        for code in original_cells
    ]
    normalized_fixed = [
        normalize_code(code, ignore_empty_lines=config.ignore_empty_lines)
        for code in fixed_cells
    ]

    aligned = [fixed_cells[i] if i < len(fixed_cells) else "" for i in range(len(original_cells))]
    matcher = SequenceMatcher(a=normalized_original, b=normalized_fixed, autojunk=False)

    for tag, orig_start, orig_end, fixed_start, fixed_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(orig_end - orig_start):
                aligned[orig_start + offset] = fixed_cells[fixed_start + offset]
            continue

        if tag == "replace":
            shared = min(orig_end - orig_start, fixed_end - fixed_start)
            for offset in range(shared):
                aligned[orig_start + offset] = fixed_cells[fixed_start + offset]

    return aligned


def classify_changes(
    reference_changes: dict[int, str],
    summary_changes: dict[int, str],
    aligned_fixed_cells: list[str],
    config: ComparisonConfig,
) -> tuple[str, str, dict[int, str]]:
    summary_cell_match_reasons: dict[int, str] = {}
    mismatched_summary_changes = []
    for cell_index, summary_code in summary_changes.items():
        fixed_code = (
            aligned_fixed_cells[cell_index]
            if cell_index < len(aligned_fixed_cells)
            else ""
        )
        reason = cell_match_reason(summary_code, fixed_code, config)
        summary_cell_match_reasons[cell_index] = reason
        if reason == "mismatch":
            mismatched_summary_changes.append(cell_index)

    if mismatched_summary_changes:
        return PLAUSIBLE, "summary_cell_mismatch", summary_cell_match_reasons

    if reference_changes.keys() == summary_changes.keys():
        return VALID, "all_summary_cells_match_change_set_equal", summary_cell_match_reasons

    return (
        VALID_WITH_EXTRA_CHANGES,
        "all_summary_cells_match_change_set_differs",
        summary_cell_match_reasons,
    )


def summarize_differences(
    reference_changes: dict[int, str],
    summary_changes: dict[int, str],
    original_cells: list[str],
    aligned_fixed_cells: list[str],
    final_cells: list[str],
    config: ComparisonConfig,
) -> dict[str, Any]:
    missing_reference_changes: list[int] = []
    extra_summary_changes: list[int] = []
    mismatched_shared_changes: list[int] = []

    reference_indices = set(reference_changes)
    summary_indices = set(summary_changes)

    for cell_index in sorted(reference_indices - summary_indices):
        missing_reference_changes.append(cell_index)

    for cell_index in sorted(summary_indices - reference_indices):
        extra_summary_changes.append(cell_index)

    for cell_index in sorted(reference_indices & summary_indices):
        if not cells_match(reference_changes[cell_index], summary_changes[cell_index], config):
            mismatched_shared_changes.append(cell_index)

    mismatch_details = []
    # Keep details focused on what the summary actually changed.
    detailed_indices = extra_summary_changes + mismatched_shared_changes
    for cell_index in detailed_indices:
        original = original_cells[cell_index] if cell_index < len(original_cells) else ""
        fixed = (
            aligned_fixed_cells[cell_index]
            if cell_index < len(aligned_fixed_cells)
            else ""
        )
        final = final_cells[cell_index] if cell_index < len(final_cells) else ""
        mismatch_details.append(
            {
                "cell_index": cell_index,
                "original": original,
                "reference_fixed": fixed,
                "summary_final": final,
            }
        )

    return {
        "missing_reference_changes": missing_reference_changes,
        "extra_summary_changes": extra_summary_changes,
        "mismatched_shared_changes": mismatched_shared_changes,
        "mismatch_details": mismatch_details,
    }


def resolve_fixed_notebook(benchmark_dir: Path, instance_name: str) -> Path:
    direct_path = benchmark_dir / instance_name / f"{instance_name}_fixed.ipynb"
    if direct_path.exists():
        return direct_path

    matches = sorted(benchmark_dir.glob(f"**/{instance_name}_fixed.ipynb"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find fixed notebook for instance '{instance_name}'")


def analyze_summary_file(
    summary_path: Path,
    results_root: Path,
    benchmark_dir: Path,
    config: ComparisonConfig,
) -> dict[str, Any] | None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = summary.get("metadata", {}).get("status", "")
    if not is_submit_status(status):
        return None

    instance_name = summary_path.stem.replace("_summary", "")
    fixed_notebook = resolve_fixed_notebook(benchmark_dir, instance_name)
    original_cells = extract_original_code_cells(summary.get("original_notebook", []))
    final_cells = apply_code_changes(original_cells, summary.get("code_changes", []) or [])
    fixed_cells = load_notebook_code_cells(fixed_notebook)
    aligned_fixed_cells = align_fixed_cells_to_original(original_cells, fixed_cells, config)
    comparison_limit = len(original_cells)

    reference_changes = compute_changed_cells(
        original_cells,
        aligned_fixed_cells,
        config,
        limit=comparison_limit,
    )
    summary_changes = compute_changed_cells(
        original_cells,
        final_cells,
        config,
        limit=comparison_limit,
    )
    classification, match_reason, summary_cell_match_reasons = classify_changes(
        reference_changes,
        summary_changes,
        aligned_fixed_cells,
        config,
    )
    difference_summary = summarize_differences(
        reference_changes,
        summary_changes,
        original_cells,
        aligned_fixed_cells,
        final_cells,
        config,
    )

    relative_path = summary_path.relative_to(results_root)
    relative_parts = relative_path.parts
    run_name = next((part for part in relative_parts if part.startswith("run_")), "unknown")
    model_name = relative_parts[1] if len(relative_parts) > 1 else "unknown"
    result_group = relative_parts[0] if relative_parts else "unknown"

    return {
        "summary_path": str(relative_path).replace("\\", "/"),
        "result_group": result_group,
        "model": model_name,
        "run": run_name,
        "instance": instance_name,
        "status": status,
        "classification": classification,
        "match_reason": match_reason,
        "summary_cell_match_reasons": {
            str(cell_index): reason
            for cell_index, reason in sorted(summary_cell_match_reasons.items())
        },
        "reference_notebook": str(fixed_notebook.relative_to(results_root.parent)).replace("\\", "/"),
        "original_code_cell_count": len(original_cells),
        "reference_code_cell_count": len(fixed_cells),
        "alignment_applied": True,
        "reference_changed_cells": sorted(reference_changes),
        "summary_changed_cells": sorted(summary_changes),
        "reference_change_count": len(reference_changes),
        "summary_change_count": len(summary_changes),
        "comparison": difference_summary,
    }


def analyze_results_directory(
    results_root: Path,
    benchmark_dir: Path,
    config: ComparisonConfig | None = None,
) -> dict[str, Any]:
    config = config or ComparisonConfig()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for summary_path in sorted(results_root.glob("**/run_*/*_summary.json")):
        try:
            record = analyze_summary_file(summary_path, results_root, benchmark_dir, config)
        except Exception as exc:
            errors.append(
                {
                    "summary_path": str(summary_path.relative_to(results_root)).replace("\\", "/"),
                    "error": str(exc),
                }
            )
            continue

        if record is not None:
            records.append(record)

    classification_counts = {
        VALID: sum(1 for record in records if record["classification"] == VALID),
        VALID_WITH_EXTRA_CHANGES: sum(
            1 for record in records if record["classification"] == VALID_WITH_EXTRA_CHANGES
        ),
        PLAUSIBLE: sum(1 for record in records if record["classification"] == PLAUSIBLE),
    }

    return {
        "metadata": {
            "results_root": str(results_root),
            "benchmark_dir": str(benchmark_dir),
            "ignore_empty_lines": config.ignore_empty_lines,
            "similarity_threshold": config.similarity_threshold,
            "similarity_function": (
                config.similarity_function.__name__
                if config.similarity_function is not None
                else None
            ),
        },
        "statistics": {
            "summaries_compared": len(records),
            "classification_counts": classification_counts,
            "errors": len(errors),
        },
        "records": records,
        "errors": errors,
    }


def main() -> None:
    report = analyze_results_directory(RESULTS_DIR, BENCHMARK_DIR)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    stats = report["statistics"]
    print(f"Compared {stats['summaries_compared']} submitted summaries")
    print(f"  {VALID}: {stats['classification_counts'][VALID]}")
    print(
        f"  {VALID_WITH_EXTRA_CHANGES}: "
        f"{stats['classification_counts'][VALID_WITH_EXTRA_CHANGES]}"
    )
    print(f"  {PLAUSIBLE}: {stats['classification_counts'][PLAUSIBLE]}")
    print(f"  Errors: {stats['errors']}")
    print(f"Wrote report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()