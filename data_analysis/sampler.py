import math
from collections import defaultdict
import json
from pathlib import Path
import random


def load_exact_matches(comparison_json_path: Path):
    """Load (run/instance) keys where classification is exactly 'Valid'."""
    if not comparison_json_path or not comparison_json_path.exists():
        return set()

    with open(comparison_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    excluded = set()
    for record in data.get("records", []):
        classification = str(record.get("classification", "")).upper()
        if classification == "VALID":
            run = record.get("run")
            instance = record.get("instance")
            if run and instance:
                excluded.add(f"{run}/{instance}")
    return excluded


def load_already_sampled_instances(sampled_json_path: Path):
    """Load already sampled instances and their existing validation labels."""
    if not sampled_json_path or not sampled_json_path.exists():
        return {}

    with open(sampled_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sampled_instances = data.get("sampled_instances", {})
    if not isinstance(sampled_instances, dict):
        return {}
    return sampled_instances


def calculate_sample_size(population_size, confidence_level=0.95, margin_error=0.05):
    """Calculate sample size for statistical sampling"""
    if population_size <= 0:
        return 0

    # Z-score for 95% confidence level
    z_score = 1.96 if confidence_level == 0.95 else 1.645

    # Formula: n = (Z^2 * p * (1-p)) / E^2
    # Using p = 0.5 for maximum variability (worst case)
    # Using p = 0.8 based on observed success rate (on 22 samples) to get a more realistic sample size
    p = 0.8
    numerator = (z_score**2) * p * (1 - p)
    denominator = margin_error**2

    # Sample size for infinite population
    n_infinite = numerator / denominator

    # Adjust for finite population
    n_adjusted = n_infinite / (1 + ((n_infinite - 1) / population_size))

    return math.ceil(n_adjusted)


def stratify_sample_on_library(instances, sample_size):
    """Stratify sample based on libraries (run_x/library_x) in success_instances, output a list of sampled instances (run_x/library_x)"""
    if sample_size <= 0 or not instances:
        return []

    if sample_size >= len(instances):
        return random.sample(list(instances), len(instances))

    def extract_library(instance):
        instance_name = instance.split("/", 1)[1] if "/" in instance else instance
        return (
            instance_name.rsplit("_", 1)[0] if "_" in instance_name else instance_name
        )

    library_buckets = defaultdict(list)
    for instance in instances:
        library_buckets[extract_library(instance)].append(instance)

    total = len(instances)
    allocations = {
        lib: min(round(len(bucket) * sample_size / total), len(bucket))
        for lib, bucket in library_buckets.items()
    }

    diff = sample_size - sum(allocations.values())
    libs_by_size = sorted(
        library_buckets, key=lambda lib: len(library_buckets[lib]), reverse=True
    )
    i = 0
    while diff != 0 and libs_by_size:
        lib = libs_by_size[i % len(libs_by_size)]
        cap = len(library_buckets[lib])
        if diff > 0 and allocations[lib] < cap:
            allocations[lib] += 1
            diff -= 1
        elif diff < 0 and allocations[lib] > 0:
            allocations[lib] -= 1
            diff += 1
        i += 1

    sampled = []
    for library, bucket in library_buckets.items():
        count = allocations[library]
        if count > 0:
            sampled.extend(random.sample(bucket, count))

    random.shuffle(sampled)
    return sampled


def main(
    target_setting: str,
    if_random_sampling=True,
    random_sampling_config=None,
    sample_size=20,
    random_seed=42,
):
    random.seed(random_seed)

    if if_random_sampling and (random_sampling_config is None):
        random_sampling_config = {"confidence_level": 0.95, "margin_error": 0.05}

    summary_path = Path(target_setting) / "overall_summary.json"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    all_success_instances = (
        summary.get("statistics", {})
        .get("outcome_distribution", {})
        .get("success_instances", [])
    )

    already_sampled_path = (
        summary_path.parent / "analysis" / "stratified_sampled_instances_labeled.json"
    )
    already_sampled_instances = load_already_sampled_instances(already_sampled_path)
    already_sampled_keys = set(already_sampled_instances.keys())

    population_size = len(all_success_instances)
    target_sample_size = sample_size
    if if_random_sampling:
        target_sample_size = calculate_sample_size(
            population_size,
            random_sampling_config["confidence_level"],
            random_sampling_config["margin_error"],
        )
        print(
            f"Calculated sample size: {target_sample_size} for population size: {population_size}"
        )
    else:
        print(
            f"Use predefined sample size: {target_sample_size} for population size: {population_size}"
        )

    # Already sampled instances are part of the final sample size.
    # Draw only the additional amount needed to reach target_sample_size.
    needed_new_samples = max(0, target_sample_size - len(already_sampled_keys))
    remaining_candidates = [
        item for item in all_success_instances if item not in already_sampled_keys
    ]
    sampled_instances = stratify_sample_on_library(
        remaining_candidates, needed_new_samples
    )

    if needed_new_samples == 0:
        print(
            "Already sampled instances already meet/exceed target sample size. No new samples drawn."
        )
    else:
        print(
            f"Already sampled: {len(already_sampled_keys)}. "
            f"Needed new: {needed_new_samples}. "
            f"Newly sampled: {len(sampled_instances)}."
        )

    def sort_key(instance):
        run_part, item_part = (
            instance.split("/", 1) if "/" in instance else ("run_0", instance)
        )
        run_num = (
            int(run_part.split("_")[-1]) if run_part.split("_")[-1].isdigit() else 0
        )
        library = item_part.rsplit("_", 1)[0] if "_" in item_part else item_part
        idx_str = item_part.rsplit("_", 1)[-1] if "_" in item_part else "0"
        item_idx = int(idx_str) if idx_str.isdigit() else 0
        return (run_num, library, item_idx, item_part)

    # Merge old labeled samples with newly sampled instances.
    # Keep old labels and initialize new labels as empty strings.
    final_sampled_instances = {
        key: value for key, value in already_sampled_instances.items()
    }
    for instance in sampled_instances:
        final_sampled_instances.setdefault(instance, "")

    sorted_instance_keys = sorted(final_sampled_instances.keys(), key=sort_key)

    fixed_comparison_path = (
        Path(target_setting) / "analysis" / "fixed_notebook_comparison.json"
    )
    exact_matches = load_exact_matches(fixed_comparison_path)

    # Auto-fill exact matches to "valid" only when no manual label exists.
    for instance in sorted_instance_keys:
        existing_label = final_sampled_instances.get(instance)
        if (
            existing_label is None or str(existing_label).strip() == ""
        ) and instance in exact_matches:
            final_sampled_instances[instance] = "valid"

    ordered_sampled_instances = {
        instance: final_sampled_instances[instance] for instance in sorted_instance_keys
    }

    output = {}
    if if_random_sampling:
        output["random_sampling_config"] = random_sampling_config
    else:
        output["random_sampling_config"] = "N/A (predefined sample size)"

    output["target_setting"] = target_setting
    output["random_seed"] = random_seed
    # output["comparison_json_path"] = str(fixed_comparison_path) if fixed_comparison_path else None
    output["population_size"] = population_size
    output["sample_size"] = len(ordered_sampled_instances)
    output["sampled_instances"] = ordered_sampled_instances

    output_dir = summary_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stratified_sampled_instances.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved stratified sample to: {output_path}")


if __name__ == "__main__":
    random_seed = 42

    if_random_sampling = True
    random_sampling_config = {"confidence_level": 0.95, "margin_error": 0.05}
    # if_random_sampling = False
    # sample_size = 20

    target_setting = "results/agent/glm-4.7-355b"

    main(
        target_setting=target_setting,
        if_random_sampling=if_random_sampling,
        random_sampling_config=random_sampling_config,
        random_seed=random_seed,
    )
