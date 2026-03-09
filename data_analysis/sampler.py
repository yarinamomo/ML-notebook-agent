
import math
from collections import defaultdict
import json
from pathlib import Path
import random

def calculate_sample_size(population_size, confidence_level=0.95, margin_error=0.05):
    """Calculate sample size for statistical sampling"""
    # Z-score for 95% confidence level
    z_score = 1.96 if confidence_level == 0.95 else 1.645
    
    # Formula: n = (Z^2 * p * (1-p)) / E^2
    # Using p = 0.5 for maximum variability (worst case)
    p = 0.5
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


def main():
    confidence_level = 0.9
    margin_error = 0.1

    target_setting = "results/baseline/glm-4.7-355b"
    summary_path = Path(target_setting) / "overall_summary.json"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    success_instances = (
        summary.get("statistics", {})
        .get("outcome_distribution", {})
        .get("success_instances", [])
    )

    population_size = len(success_instances)
    sample_size = calculate_sample_size(population_size, confidence_level, margin_error)
    print(f"Calculated sample size: {sample_size} for population size: {population_size}")

    sampled_instances = stratify_sample_on_library(success_instances, sample_size)

    def sort_key(instance):
        run_part, item_part = instance.split("/", 1) if "/" in instance else ("run_0", instance)
        run_num = int(run_part.split("_")[-1]) if run_part.split("_")[-1].isdigit() else 0
        library = item_part.rsplit("_", 1)[0] if "_" in item_part else item_part
        idx_str = item_part.rsplit("_", 1)[-1] if "_" in item_part else "0"
        item_idx = int(idx_str) if idx_str.isdigit() else 0
        return (run_num, library, item_idx, item_part)

    sampled_instances = sorted(sampled_instances, key=sort_key)

    output = {
        "target_setting": target_setting,
        "confidence_level": confidence_level,
        "margin_error": margin_error,
        "population_size": population_size,
        "sample_size": sample_size,
        "sampled_instances": sampled_instances,
    }

    output_dir = summary_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stratified_sampled_instances.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved stratified sample to: {output_path}")

if __name__ == "__main__":
    main()