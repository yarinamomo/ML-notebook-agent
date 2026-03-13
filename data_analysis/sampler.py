import math
from collections import defaultdict
import json
from pathlib import Path
import random


def load_valid_exclusion_pairs(comparison_json_path: Path):
    """Load (run/instance) keys where classification is exactly 'Valid'."""
    if not comparison_json_path or not comparison_json_path.exists():
        return set()

    with open(comparison_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    excluded = set()
    for record in data.get("records", []):
        if (record.get("classification")).upper() == "VALID":
            run = record.get("run")
            instance = record.get("instance")
            if run and instance:
                excluded.add(f"{run}/{instance}")
    return excluded


def infer_comparison_json_path(target_setting: str) -> Path | None:
    """Infer fixed_notebook_comparison.json path from target_setting/model."""
    target_path = Path(target_setting)

    preferred = target_path / "analysis" / "fixed_notebook_comparison.json"
    if preferred.exists():
        return preferred
    else:
        print(f"Fixed comparison JSON not found at: {preferred}")
        return None

def calculate_sample_size(population_size, confidence_level=0.95, margin_error=0.05):
    """Calculate sample size for statistical sampling"""
    # Z-score for 95% confidence level
    z_score = 1.96 if confidence_level == 0.95 else 1.645
    
    # Formula: n = (Z^2 * p * (1-p)) / E^2
    # Using p = 0.5 for maximum variability (worst case)
    # Using p = 0.8 based on observed success rate (on 22 samples) to get a more realistic sample size
    p = 0.8
    numerator = (z_score ** 2) * p * (1 - p)
    denominator = margin_error ** 2
    
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
        return instance_name.rsplit("_", 1)[0] if "_" in instance_name else instance_name

    library_buckets = defaultdict(list)
    for instance in instances:
        library_buckets[extract_library(instance)].append(instance)

    total = len(instances)
    allocations = {
        lib: min(round(len(bucket) * sample_size / total), len(bucket))
        for lib, bucket in library_buckets.items()
    }

    diff = sample_size - sum(allocations.values())
    libs_by_size = sorted(library_buckets, key=lambda lib: len(library_buckets[lib]), reverse=True)
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


def main(target_setting: str, if_random_sampling = True, random_sampling_config = None,
         sample_size = 20, random_seed = 42):

    random.seed(random_seed)

    if if_random_sampling and (random_sampling_config is None):
        random_sampling_config = {
            "confidence_level": 0.95,
            "margin_error": 0.05
        }

    summary_path = Path(target_setting) / "overall_summary.json"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    success_instances = (
        summary.get("statistics", {})
        .get("outcome_distribution", {})
        .get("success_instances", [])
    )

    fixed_comparison_path = infer_comparison_json_path(target_setting)

    excluded_valid_pairs = load_valid_exclusion_pairs(fixed_comparison_path) if fixed_comparison_path else set()
    if excluded_valid_pairs:
        original_population_size = len(success_instances)
        success_instances = [
            item for item in success_instances
            if item not in excluded_valid_pairs
        ]
        print(
            "Excluded "
            f"{original_population_size - len(success_instances)} 'Valid' cases "
            f"from {fixed_comparison_path}"
        )
    elif fixed_comparison_path is None:
        print("No fixed_notebook_comparison.json found automatically; skipping 'Valid' exclusions.")

    population_size = len(success_instances)
    if if_random_sampling:
        sample_size = calculate_sample_size(population_size, random_sampling_config["confidence_level"], random_sampling_config["margin_error"])
        print(f"Calculated sample size: {sample_size} for population size: {population_size}")
    else:
        print(f"Use predefined sample size: {sample_size} for population size: {population_size}")
    sampled_instances = stratify_sample_on_library(success_instances, sample_size)

    def sort_key(instance):
        run_part, item_part = instance.split("/", 1) if "/" in instance else ("run_0", instance)
        run_num = int(run_part.split("_")[-1]) if run_part.split("_")[-1].isdigit() else 0
        library = item_part.rsplit("_", 1)[0] if "_" in item_part else item_part
        idx_str = item_part.rsplit("_", 1)[-1] if "_" in item_part else "0"
        item_idx = int(idx_str) if idx_str.isdigit() else 0
        return (run_num, library, item_idx, item_part)

    sampled_instances = sorted(sampled_instances, key=sort_key)

    output = {}
    if if_random_sampling:
        output["random_sampling_config"] = random_sampling_config
    else:
        output["random_sampling_config"] = "N/A (predefined sample size)"

    output["target_setting"] = target_setting
    output["random_seed"] = random_seed
    output["comparison_json_path"] = str(fixed_comparison_path) if fixed_comparison_path else None
    output["excluded_valid_pairs_count"] = len(excluded_valid_pairs)
    output["population_size"] = population_size
    output["sample_size"] = sample_size
    output["sampled_instances"] = dict.fromkeys(sampled_instances, "")

    output_dir = summary_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stratified_sampled_instances.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved stratified sample to: {output_path}")

if __name__ == "__main__":
    random_seed = 42
    
    if_random_sampling = True
    random_sampling_config = {
        "confidence_level": 0.9,
        "margin_error": 0.1
    }
    # if_random_sampling = False
    # sample_size = 20
    
    target_setting = "results/baseline_without_cell_outputs/glm-4.7-355b"

    main(target_setting = target_setting, if_random_sampling = if_random_sampling, random_sampling_config = random_sampling_config, random_seed = random_seed)