"""
Summarize LLM inference cost per model across all settings and runs.

Reads ``info.model_stats.instance_cost`` from every ``*.traj.json`` file under
``<root>/<setting>/<model>/run_*/``. The baseline settings write trajectories
without an ``info`` block, so for those the cost is taken from the sibling
``<instance>_summary.json`` (``metadata.cost``, the same number rounded to 4
decimals). Reports, per model:

* per-instance cost statistics (mean, std, median, min/max) over all
  settings x runs,
* total cost,
* run-to-run variation of the per-run totals,
* the same breakdown per setting.

Several result roots can be given at once: each is reported on its own and
then all of them are pooled into a combined report.

Usage:
    python -m data_analysis.summarize_inference_cost
    python -m data_analysis.summarize_inference_cost --root results_JunoBench results_new
    python -m data_analysis.summarize_inference_cost --root results_new --csv costs.csv
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOTS = [Path("results_JunoBench"), Path("results_new")]

SETTING_LABEL_MAP = {
    'baseline_with_all_outputs': 'SSB+CO',
    'baseline_without_cell_outputs': 'SSB',
    'baseline_without_all_outputs': 'SSB-ERR',
    'agent_without_run_code_and_cell_outputs': 'Agent-lite',
    'agent': 'Agent-full',
}

LLM_ABBREVIATIONS = {
    'glm-4.7-355b': 'GLM',
    'gpt-5.3-codex': 'codex',
}


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [warn] could not read {path}: {exc}")
        return None


def read_instance_cost(traj_path):
    """Return (cost, source) for a *.traj.json file, or (None, None).

    Agent trajectories store the cost in info.model_stats.instance_cost.
    Baseline trajectories have no info block, so fall back to the sibling
    <instance>_summary.json, whose metadata.cost holds the same value
    rounded to 4 decimals.
    """
    data = _load_json(traj_path)
    if data is not None:
        cost = data.get("info", {}).get("model_stats", {}).get("instance_cost")
        if cost is not None:
            return float(cost), "traj"

    summary_path = traj_path.with_name(traj_path.name[:-len(".traj.json")] + "_summary.json")
    if summary_path.exists():
        summary = _load_json(summary_path)
        if summary is not None:
            cost = summary.get("metadata", {}).get("cost")
            if cost is not None:
                return float(cost), "summary"

    print(f"  [warn] no cost found for {traj_path}")
    return None, None


def collect_costs(root, settings=None, models=None):
    """Collect per-instance costs.

    Returns a list of records:
    {root, setting, model, run, instance, cost, source}.
    """
    settings = settings or list(SETTING_LABEL_MAP)
    models = models or list(LLM_ABBREVIATIONS)
    records = []

    for setting in settings:
        for model in models:
            model_dir = root / setting / model
            if not model_dir.is_dir():
                print(f"  [skip] missing directory {model_dir}")
                continue
            run_dirs = sorted(
                (d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("run_")),
                key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else 0,
            )
            if not run_dirs:
                print(f"  [skip] no run_* folders in {model_dir}")
                continue
            for run_dir in run_dirs:
                try:
                    run_num = int(run_dir.name.split("_")[-1])
                except ValueError:
                    run_num = 0
                for traj_path in sorted(run_dir.glob("*.traj.json")):
                    cost, source = read_instance_cost(traj_path)
                    if cost is None:
                        continue
                    records.append({
                        "setting": setting,
                        "model": model,
                        "run": run_num,
                        "instance": traj_path.name[:-len(".traj.json")],
                        "cost": cost,
                        "source": source,
                        "root": root.name,
                    })
    return records


def describe(costs):
    """Basic statistics for a list of per-instance costs."""
    n = len(costs)
    if n == 0:
        return None
    mean = statistics.fmean(costs)
    return {
        "n": n,
        "total": sum(costs),
        "mean": mean,
        "std": statistics.stdev(costs) if n > 1 else 0.0,   # sample std
        "sem": (statistics.stdev(costs) / (n ** 0.5)) if n > 1 else 0.0,
        "cv": (statistics.stdev(costs) / mean) if n > 1 and mean else 0.0,
        "median": statistics.median(costs),
        "min": min(costs),
        "max": max(costs),
    }


def run_totals(records):
    """Total cost per (setting, run), used for run-to-run variation."""
    totals = defaultdict(float)
    for r in records:
        totals[(r["setting"], r["run"])] += r["cost"]
    return totals


def _fmt_row(label, s, width=30):
    return (f"  {label:<{width}} {s['n']:>5} {s['total']:>10.2f} {s['mean']:>9.4f} "
            f"{s['std']:>9.4f} {s['cv'] * 100:>7.1f}% {s['median']:>9.4f} "
            f"{s['min']:>9.4f} {s['max']:>9.4f}")


def _fmt_header(title, width=30):
    return (f"  {title:<{width}} {'N':>5} {'total($)':>10} {'mean($)':>9} "
            f"{'std($)':>9} {'CV':>8} {'median':>9} {'min':>9} {'max':>9}")


def report(records, title):
    """Print the full per-model breakdown for one pool of records."""
    print("\n\n" + "#" * 110)
    print(f"### {title}")
    print("#" * 110)

    if not records:
        print("No trajectories found.")
        return 0.0

    roots = sorted({r["root"] for r in records})
    if len(roots) > 1:
        print(f"  Pooled roots: {', '.join(roots)}")

    models = sorted({r["model"] for r in records}, key=lambda m: list(LLM_ABBREVIATIONS).index(m)
                    if m in LLM_ABBREVIATIONS else 99)
    grand_total = 0.0

    for model in models:
        model_records = [r for r in records if r["model"] == model]
        label = LLM_ABBREVIATIONS.get(model, model)
        costs = [r["cost"] for r in model_records]
        overall = describe(costs)
        grand_total += overall["total"]

        print("\n" + "=" * 110)
        print(f"MODEL: {model}  ({label})")
        print("=" * 110)

        # Per-setting breakdown.
        print(_fmt_header("per setting"))
        for setting in SETTING_LABEL_MAP:
            setting_costs = [r["cost"] for r in model_records if r["setting"] == setting]
            s = describe(setting_costs)
            if s is None:
                continue
            print(_fmt_row(f"{SETTING_LABEL_MAP[setting]} ({setting[:14]})", s))
        print("  " + "-" * 106)
        print(_fmt_row("ALL SETTINGS", overall))

        # Run-to-run variation: total cost of each (setting, run).
        totals = run_totals(model_records)
        per_run = defaultdict(list)
        for (setting, run), total in totals.items():
            per_run[setting].append((run, total))
        print("\n  Run-to-run variation (total cost per run, $):")
        print(f"    {'setting':<26} " + " ".join(f"{'run ' + str(i):>10}" for i in (1, 2, 3))
              + f" {'mean':>10} {'std':>9} {'CV':>7}")
        for setting in SETTING_LABEL_MAP:
            if setting not in per_run:
                continue
            runs = dict(per_run[setting])
            values = [runs[k] for k in sorted(runs)]
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            cells = " ".join(f"{runs.get(i, float('nan')):>10.2f}" for i in sorted(runs))
            print(f"    {SETTING_LABEL_MAP[setting]:<26} {cells} {mean:>10.2f} {std:>9.2f} "
                  f"{(std / mean * 100) if mean else 0:>6.1f}%")

        n_rounded = sum(1 for r in model_records if r.get("source") == "summary")
        if n_rounded:
            print(f"\n  Note: {n_rounded}/{len(model_records)} costs came from *_summary.json "
                  f"(metadata.cost, rounded to 4 decimals) because those trajectories "
                  f"carry no instance_cost.")

        if len(roots) > 1:
            print("\n  Contribution of each result root, per setting:")
            for root in roots:
                s = describe([r["cost"] for r in model_records if r["root"] == root])
                if s is None:
                    continue
                print(f"    {root}: N={s['n']}  total=${s['total']:.2f}  "
                      f"mean=${s['mean']:.4f}  std=${s['std']:.4f}")
                for setting in SETTING_LABEL_MAP:
                    sub = describe([r["cost"] for r in model_records
                                    if r["root"] == root and r["setting"] == setting])
                    if sub is None:
                        continue
                    print(f"      {SETTING_LABEL_MAP[setting]:<14} N={sub['n']:>5}  "
                          f"total=${sub['total']:>7.2f}  mean=${sub['mean']:.4f}  "
                          f"std=${sub['std']:.4f}  median=${sub['median']:.4f}")

        n_settings = len({r["setting"] for r in model_records})
        n_runs = len({(r["setting"], r["run"]) for r in model_records})
        n_instances = len({(r["setting"], r["run"], r["instance"]) for r in model_records})
        print(f"\n  Coverage: {n_settings} settings, {n_runs} setting-run pairs, "
              f"{n_instances} trajectories")
        print(f"  Average cost per instance: ${overall['mean']:.4f} "
              f"(+/- {overall['std']:.4f} std, +/- {overall['sem']:.4f} sem)")
        print(f"  Total cost: ${overall['total']:.2f}")

    print_setting_table(records, models)

    print("\n" + "=" * 110)
    print(f"GRAND TOTAL across all models/settings/runs: ${grand_total:.2f}")
    print("=" * 110)
    return grand_total


def print_setting_table(records, models):
    """Setting-first view: one row per setting, one column group per model."""
    print("\n" + "=" * 110)
    print("PER SETTING (all models side by side)")
    print("=" * 110)

    header = f"  {'setting':<12}"
    for model in models:
        header += f" | {LLM_ABBREVIATIONS.get(model, model):^43}"
    print(header)
    sub = f"  {'':<12}"
    for _ in models:
        sub += f" | {'N':>5} {'total($)':>9} {'mean($)':>9} {'std($)':>9} {'med($)':>7}"
    print(sub)
    print("  " + "-" * 108)

    for setting in SETTING_LABEL_MAP:
        row = f"  {SETTING_LABEL_MAP[setting]:<12}"
        any_data = False
        for model in models:
            s = describe([r["cost"] for r in records
                          if r["model"] == model and r["setting"] == setting])
            if s is None:
                row += f" | {'-':>5} {'-':>9} {'-':>9} {'-':>9} {'-':>7}"
                continue
            any_data = True
            row += (f" | {s['n']:>5} {s['total']:>9.2f} {s['mean']:>9.4f} "
                    f"{s['std']:>9.4f} {s['median']:>7.4f}")
        if any_data:
            print(row)

    print("  " + "-" * 108)
    total_row = f"  {'ALL':<12}"
    for model in models:
        s = describe([r["cost"] for r in records if r["model"] == model])
        if s is None:
            total_row += f" | {'-':>5} {'-':>9} {'-':>9} {'-':>9} {'-':>7}"
            continue
        total_row += (f" | {s['n']:>5} {s['total']:>9.2f} {s['mean']:>9.4f} "
                      f"{s['std']:>9.4f} {s['median']:>7.4f}")
    print(total_row)


ROOT_SHORT_NAMES = {
    "results_JunoBench": "JunoBench",
    "results_new": "new",
}

FLAT_FIELDS = ["scope", "n_glm", "n_codex", "glm_total", "glm_mean", "glm_std", "glm_median",
               "codex_total", "codex_mean", "codex_std", "codex_median"]


def flat_rows(records, glm="glm-4.7-355b", codex="gpt-5.3-codex"):
    """One row per (root x setting) scope, with both models side by side.

    The instance count is reported per model because coverage can differ:
    results_new has no codex runs for two of the baseline settings.
    """
    roots = sorted({r["root"] for r in records})
    scopes = roots + ([None] if len(roots) > 1 else [])   # None = all roots pooled
    rows = []

    for root in scopes:
        root_label = ROOT_SHORT_NAMES.get(root, root) if root else "COMBINED"
        pool = records if root is None else [r for r in records if r["root"] == root]
        for setting in list(SETTING_LABEL_MAP) + [None]:   # None = all settings
            setting_label = SETTING_LABEL_MAP.get(setting, "ALL")
            subset = pool if setting is None else [r for r in pool if r["setting"] == setting]
            g = describe([r["cost"] for r in subset if r["model"] == glm])
            c = describe([r["cost"] for r in subset if r["model"] == codex])
            if g is None and c is None:
                continue
            row = {"scope": f"{root_label} / {setting_label}",
                   "n_glm": g["n"] if g else 0,
                   "n_codex": c["n"] if c else 0}
            for prefix, s in (("glm", g), ("codex", c)):
                for key in ("total", "mean", "std", "median"):
                    row[f"{prefix}_{key}"] = s[key] if s else None
            rows.append(row)
    return rows


def print_flat_table(records):
    """Flat scope table: scope, per-model N, total, mean, std, median."""
    rows = flat_rows(records)
    if not rows:
        return

    def fmt(value, key):
        if value is None:
            return "-"
        if key == "scope" or key.startswith("n_"):
            return str(value)
        return f"{value:.4f}"

    printed = [[fmt(r[k], k) for k in FLAT_FIELDS] for r in rows]
    widths = [max(len(FLAT_FIELDS[i]), max(len(p[i]) for p in printed))
              for i in range(len(FLAT_FIELDS))]

    print("\n\n" + "=" * 110)
    print("FLAT SUMMARY (scope x model)")
    print("=" * 110)
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(FLAT_FIELDS)))
    print("-+-".join("-" * w for w in widths))
    for p in printed:
        print(" | ".join(v.ljust(widths[i]) if i == 0 else v.rjust(widths[i])
                         for i, v in enumerate(p)))


def write_flat_csv(records, csv_path):
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FLAT_FIELDS)
        writer.writeheader()
        writer.writerows(flat_rows(records))
    print(f"Wrote flat summary CSV to {csv_path}")


def write_csv(records, csv_path):
    """Write per-(root, model, setting) summary rows, plus pooled rows per model."""
    import csv

    fields = ["root", "model", "model_label", "setting", "setting_label", "n",
              "total", "mean", "std", "sem", "cv", "median", "min", "max"]
    rows = []
    roots = sorted({r["root"] for r in records})
    # One block per root, then an "ALL_ROOTS" block pooling them.
    root_scopes = roots + (["ALL_ROOTS"] if len(roots) > 1 else [])
    for root in root_scopes:
        scope = records if root == "ALL_ROOTS" else [r for r in records if r["root"] == root]
        for model in sorted({r["model"] for r in scope}):
            model_records = [r for r in scope if r["model"] == model]
            for setting in list(SETTING_LABEL_MAP) + ["ALL"]:
                if setting == "ALL":
                    subset = model_records
                else:
                    subset = [r for r in model_records if r["setting"] == setting]
                s = describe([r["cost"] for r in subset])
                if s is None:
                    continue
                rows.append({
                    "root": root,
                    "model": model,
                    "model_label": LLM_ABBREVIATIONS.get(model, model),
                    "setting": setting,
                    "setting_label": SETTING_LABEL_MAP.get(setting, "ALL"),
                    **{k: s[k] for k in ("n", "total", "mean", "std", "sem",
                                         "cv", "median", "min", "max")},
                })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote summary CSV to {csv_path}")


def write_per_instance_csv(records, csv_path):
    import csv

    fields = ["root", "model", "setting", "run", "instance", "cost", "source"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote per-instance CSV to {csv_path}")


def main(roots=None, settings=None, models=None, csv_path=None, per_instance_csv=None,
         combined=True, flat_only=False, flat_csv=None):
    """Report each root on its own, then all roots pooled together."""
    roots = [Path(r) for r in (roots or DEFAULT_ROOTS)]

    all_records = []
    for root in roots:
        if not flat_only:
            print(f"\nCollecting instance costs from {root} ...")
        records = collect_costs(root, settings, models)
        if not flat_only:
            report(records, f"{root}  (on its own)")
        all_records.extend(records)

    if combined and len(roots) > 1 and not flat_only:
        report(all_records, "COMBINED: " + " + ".join(str(r) for r in roots))

    print_flat_table(all_records)

    if csv_path:
        write_csv(all_records, csv_path)
    if flat_csv:
        write_flat_csv(all_records, flat_csv)
    if per_instance_csv:
        write_per_instance_csv(all_records, per_instance_csv)
    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", nargs="+", default=[str(r) for r in DEFAULT_ROOTS],
                        help="One or more results root directories. Each is reported "
                             "separately and then all are pooled "
                             f"(default: {' '.join(str(r) for r in DEFAULT_ROOTS)})")
    parser.add_argument("--settings", nargs="+", default=None,
                        help="Settings to include (default: all 5)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to include (default: all known)")
    parser.add_argument("--no-combined", action="store_true",
                        help="Skip the pooled report over all roots")
    parser.add_argument("--flat", action="store_true",
                        help="Print only the flat scope x model table")
    parser.add_argument("--csv", default=None, help="Write the summary table to this CSV file")
    parser.add_argument("--flat-csv", default=None,
                        help="Write the flat scope x model table to this CSV file")
    parser.add_argument("--per-instance-csv", default=None,
                        help="Write every per-instance cost to this CSV file")
    args = parser.parse_args()
    main(args.root, args.settings, args.models, args.csv, args.per_instance_csv,
         combined=not args.no_combined, flat_only=args.flat, flat_csv=args.flat_csv)
