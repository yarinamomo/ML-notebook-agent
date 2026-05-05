"""
Analyze run_code operations from trajectory files to identify patterns in custom Python code.

This script extracts all run_code() operations from trajectory JSON files and analyzes
the patterns in the custom code that was executed.
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import re
from itertools import combinations

def extract_run_code_operations(traj_file):
    """
    Extract all run_code operations from a trajectory file.
    
    Args:
        traj_file: Path to the trajectory JSON file
    
    Returns:
        List of dictionaries containing code and metadata
    """
    try:
        with open(traj_file, 'r', encoding='utf-8', errors='ignore') as f:
            traj_data = json.load(f)
        
        run_code_ops = []
        
        if 'messages' in traj_data:
            for msg in traj_data['messages']:
                if msg.get('role') == 'assistant' and 'extra' in msg:
                    actions = msg['extra'].get('actions', [])
                    for action in actions:
                        if action.get('tool_name') == 'run_code':
                            code = action.get('arguments', {}).get('code', '')
                            if code:
                                run_code_ops.append({
                                    'code': code,
                                    'command': action.get('command', ''),
                                })
        
        return run_code_ops
    
    except Exception as e:
        print(f"Error processing {traj_file}: {e}")
        return []


def categorize_code(code):
    """
    Categorize a code snippet based on its content.
    
    Returns:
        List of categories that apply to this code
    """
    categories = []
    
    # Print/display output
    if 'print(' in code:
        categories.append('prints')

    # # File/directory/CSV/data files operations
    # if any(pattern in code for pattern in ['os.listdir', 'os.path.exists', 'os.getcwd', 'glob.glob', 'pd.read_csv', 'pd.read_', 'np.load']):
    #     categories.append('file_exploration')

    # # code complexity
    # # Error handling/debugging
    # if any(pattern in code for pattern in ['try:', 'except:', 'assert', 'raise']):
    #     categories.append('error_handling')

    # # function definition
    # if re.search(r'^\s*def\s+\w+\s*\(', code, re.MULTILINE):
    #     categories.append('function_definition')

    # # class definition
    # if re.search(r'^\s*class\s+\w+\s*(\(\w+\))?:', code, re.MULTILINE):
    #     categories.append('class_definition')

    # runtime information categories
    # structural information (size, shape, count, dimensions)
    if any(pattern in code for pattern in ['.shape', 'len(', '.dtypes', '.ndim', 'size']):
        categories.append('structure')

    # type semantics (type, dtype, shcema-level properties)
    if any(pattern in code for pattern in ['type(', '.dtypes', 'isinstance(', 'dtype']):
        categories.append('type')

    # value semantics (concrete values: value range, presence of NaNs, number of classes, example values)
    if (
        any(pattern in code for pattern in ['.head(', '.tail(', '.sample(', '.info(', '.describe(', '.columns', '.index', '.values', '.keys', '.listdir(', '.class_names', '.iloc', '.loc']) 
        or ('in' in code and 'columns' in code)
        or any(re.search(pattern, code) for pattern in [r'\.read\(\d+\)', r'\[\s*-?\d*\s*:\s*-?\d*\s*\]'])
    ):
        categories.append('value')
    
    # # DataFrame operations
    # if any(pattern in code for pattern in ['.iloc', '.loc', '.query(', '.groupby(']):
    #     categories.append('dataframe_operations')
    # # Imports
    # if re.search(r'^\s*(import|from)\s+', code, re.MULTILINE):
    #     categories.append('imports')
    
    if not categories:
        categories.append('other')
    
    return categories


def extract_key_operations(code):
    """
    Extract key operations/patterns from code.
    
    Returns:
        Dictionary with operation details
    """
    operations = {
        'imports': [],
        'methods': [],
        'print_statements': 0,
        'variables_printed': [],
    }
    
    # Extract imports
    import_matches = re.findall(r'(?:^|\n)\s*(?:import|from)\s+(\S+)', code)
    operations['imports'] = import_matches
    
    # Extract methods
    methods = re.findall(r'\.(\w+)\s*\(', code)
    operations['methods'] = list(set(methods))
    
    # Count print statements
    operations['print_statements'] = code.count('print(')
    
    # Extract variable names being printed
    var_prints = re.findall(r'print\(["\'].*?["\'],?\s*(\w+)', code)
    operations['variables_printed'] = var_prints
    
    return operations


def analyze_all_runs(base_dir):
    """
    Analyze all run_code operations across all runs.
    
    Args:
        base_dir: Base directory containing run folders
    
    Returns:
        Dictionary with analysis results
    """
    base_path = Path(base_dir)
    
    all_run_code_ops = []
    category_counts = Counter()
    import_counts = Counter()
    method_counts = Counter()
    
    # Process each run
    for run_num in [1, 2, 3]:
        run_dir = base_path / f'run_{run_num}'
        
        if not run_dir.exists():
            continue
        
        print(f"\nProcessing {run_dir}...")
        
        # Find all trajectory files
        traj_files = list(run_dir.glob('*.traj.json'))
        print(f"  Found {len(traj_files)} trajectory files")
        
        run_ops_count = 0
        for traj_file in traj_files:
            ops = extract_run_code_operations(traj_file)
            run_ops_count += len(ops)
            
            for op in ops:
                code = op['code']
                
                # Categorize
                categories = categorize_code(code)
                for cat in categories:
                    category_counts[cat] += 1
                
                # Extract key operations
                key_ops = extract_key_operations(code)
                
                # Count imports
                for imp in key_ops['imports']:
                    import_counts[imp] += 1
                
                # Count methods
                for method in key_ops['methods']:
                    method_counts[method] += 1
                
                # Store operation with metadata
                all_run_code_ops.append({
                    'run': run_num,
                    'file': traj_file.stem,
                    'code': code,
                    'categories': categories,
                    'key_operations': key_ops,
                })
        
        print(f"  Extracted {run_ops_count} run_code operations")
    
    return {
        'operations': all_run_code_ops,
        'category_counts': category_counts,
        'import_counts': import_counts,
        'method_counts': method_counts,
    }


def generate_report(analysis_results, output_file):
    """
    Generate a comprehensive report of the analysis.
    
    Args:
        analysis_results: Results from analyze_all_runs
        output_file: Path to save the report
    """
    operations = analysis_results['operations']
    category_counts = analysis_results['category_counts']
    import_counts = analysis_results['import_counts']
    method_counts = analysis_results['method_counts']
    
    # Compute overlap stats for report
    overlaps = compute_category_overlaps(operations)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RUN_CODE OPERATIONS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        f.write(f"Total run_code operations: {len(operations)}\n\n")
        
        # Category breakdown
        f.write("=" * 80 + "\n")
        f.write("CATEGORY BREAKDOWN\n")
        f.write("=" * 80 + "\n")
        for category, count in category_counts.most_common():
            percentage = (count / len(operations)) * 100
            f.write(f"{category:.<40} {count:>5} ({percentage:>5.1f}%)\n")
        f.write("\n")
        
        # Category overlap statistics
        f.write("=" * 80 + "\n")
        f.write("CATEGORY OVERLAP ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Single category operations
        f.write("Categories appearing alone (no combinations):\n")
        if overlaps['category_alone']:
            for cat in sorted(overlaps['all_categories']):
                count = overlaps['category_alone'].get(cat, 0)
                if count > 0:
                    f.write(f"  {cat:.<35} {count:>5}\n")
        else:
            f.write("  (None)\n")
        f.write("\n")
        
        # 2-way overlaps
        f.write("2-way category combinations (pairs):\n")
        pair_list = sorted(overlaps['co_occurrence_pairs'].items(), key=lambda x: x[1], reverse=True)
        for i, (pair, count) in enumerate(pair_list[:10], 1):
            f.write(f"  {i:2d}. {pair:.<45} {count:>5}\n")
        f.write(f"  ... ({len(overlaps['co_occurrence_pairs'])} total pairs)\n\n")
        
        # 3-way overlaps
        f.write("3-way category combinations (triplets):\n")
        if overlaps['co_occurrence_triplets']:
            triplet_list = sorted(overlaps['co_occurrence_triplets'].items(), key=lambda x: x[1], reverse=True)
            for i, (triplet, count) in enumerate(triplet_list[:10], 1):
                f.write(f"  {i:2d}. {triplet:.<45} {count:>5}\n")
            f.write(f"  ... ({len(overlaps['co_occurrence_triplets'])} total triplets)\n")
        else:
            f.write("  (None)\n")
        f.write("\n")
        
        # 4-way overlaps
        f.write("4-way category combinations (quads):\n")
        if overlaps['co_occurrence_quads']:
            quad_list = sorted(overlaps['co_occurrence_quads'].items(), key=lambda x: x[1], reverse=True)
            for i, (quad, count) in enumerate(quad_list[:10], 1):
                f.write(f"  {i:2d}. {quad:.<45} {count:>5}\n")
            f.write(f"  ... ({len(overlaps['co_occurrence_quads'])} total quads)\n")
        else:
            f.write("  (None)\n")
        f.write("\n")
        
        # Multi-category summary
        cat_counts = Counter(overlaps['category_counts_per_operation'])
        f.write("Operations by number of categories:\n")
        for n_cats in sorted(cat_counts.keys()):
            count = cat_counts[n_cats]
            pct = (count / len(operations)) * 100
            f.write(f"  {n_cats} category/categories: {count:>5} ({pct:>5.1f}%)\n")
        f.write("\n")
        
        # Top imports
        f.write("=" * 80 + "\n")
        f.write("TOP IMPORTS (in run_code operations)\n")
        f.write("=" * 80 + "\n")
        for imp, count in import_counts.most_common(20):
            f.write(f"{imp:.<40} {count:>5}\n")
        f.write("\n")
        
        # Top methods
        f.write("=" * 80 + "\n")
        f.write("TOP METHODS\n")
        f.write("=" * 80 + "\n")
        for method, count in method_counts.most_common(20):
            f.write(f"{method:.<40} {count:>5}\n")
        f.write("\n")
        
        # Example codes for each major category
        f.write("=" * 80 + "\n")
        f.write("EXAMPLE CODES BY CATEGORY\n")
        f.write("=" * 80 + "\n\n")
        
        # Group by category
        codes_by_category = defaultdict(list)
        for op in operations:
            for cat in op['categories']:
                if len(codes_by_category[cat]) < 5:  # Limit to 5 examples per category
                    codes_by_category[cat].append(op['code'])
        
        for category in sorted(codes_by_category.keys()):
            f.write(f"\n{'-' * 80}\n")
            f.write(f"Category: {category}\n")
            f.write(f"{'-' * 80}\n")
            for i, code in enumerate(codes_by_category[category][:3], 1):
                f.write(f"\nExample {i}:\n")
                f.write("-" * 40 + "\n")
                f.write(code)
                f.write("\n" + "-" * 40 + "\n")
        
        # Common patterns
        f.write("\n" + "=" * 80 + "\n")
        f.write("COMMON PATTERNS\n")
        f.write("=" * 80 + "\n\n")
        
        # Analyze code loc
        code_loc = [len(op['code'].splitlines()) for op in operations]
        if code_loc:
            avg_loc = sum(code_loc) / len(code_loc)
            f.write(f"Average code loc: {avg_loc:.1f} lines\n")
            f.write(f"Shortest code: {min(code_loc)} lines\n")
            f.write(f"Longest code: {max(code_loc)} lines\n\n")
        else:
            f.write("No operations found to analyze.\n\n")
        
        # Multi-category operations
        if operations:
            multi_cat_ops = [op for op in operations if len(op['categories']) > 1]
            f.write(f"Operations with multiple categories: {len(multi_cat_ops)} ({len(multi_cat_ops)/len(operations)*100:.1f}%)\n")
    
    print(f"\nReport saved to: {output_file}")


def compute_category_overlaps(operations):
    """
    Compute co-occurrence matrix for categories including 2-way, 3-way, 4+ way overlaps,
    and single category counts (no combinations).
    
    Args:
        operations: List of operations with categories
    
    Returns:
        Dictionary with overlap statistics
    """
    # Count how many categories each operation has
    category_counts_per_op = [len(op['categories']) for op in operations]
    
    # Co-occurrence tracking for different overlap types
    category_alone = Counter()  # Categories appearing alone (no combinations)
    category_pairs = Counter()  # 2-way overlaps
    category_triplets = Counter()  # 3-way overlaps
    category_quads = Counter()  # 4-way overlaps
    all_categories = set()
    
    for op in operations:
        cats = op['categories']
        all_categories.update(cats)
        
        # Count categories appearing alone (only 1 category in operation)
        if len(cats) == 1:
            category_alone[cats[0]] += 1
        
        # Count pairs of categories that appear together (2-way)
        if len(cats) >= 2:
            for cat1, cat2 in combinations(sorted(cats), 2):
                category_pairs[(cat1, cat2)] += 1
        
        # Count triplets (3-way overlaps)
        if len(cats) >= 3:
            for cat1, cat2, cat3 in combinations(sorted(cats), 3):
                category_triplets[(cat1, cat2, cat3)] += 1
        
        # Count quads and higher (4-way overlaps)
        if len(cats) >= 4:
            for cat1, cat2, cat3, cat4 in combinations(sorted(cats), 4):
                category_quads[(cat1, cat2, cat3, cat4)] += 1
    
    return {
        'category_counts_per_operation': category_counts_per_op,
        'category_alone': {cat: count for cat, count in category_alone.items()},
        'co_occurrence_pairs': {f"{cat1}|{cat2}": count for (cat1, cat2), count in category_pairs.items()},
        'co_occurrence_triplets': {f"{cat1}|{cat2}|{cat3}": count for (cat1, cat2, cat3), count in category_triplets.items()},
        'co_occurrence_quads': {f"{cat1}|{cat2}|{cat3}|{cat4}": count for (cat1, cat2, cat3, cat4), count in category_quads.items()},
        'all_categories': sorted(all_categories)
    }


def save_all_codes_json(analysis_results, output_file):
    """
    Save all extracted codes to a JSON file for further analysis.
    
    Args:
        analysis_results: Results from analyze_all_runs
        output_file: Path to save the JSON
    """
    # Create a simplified version for JSON export
    export_data = []
    
    for op in analysis_results['operations']:
        export_data.append({
            'run': op['run'],
            'file': op['file'],
            'code': op['code'],
            'categories': op['categories'],
            'imports': op['key_operations']['imports'],
            'methods': op['key_operations']['methods'],
            'print_count': op['key_operations']['print_statements'],
        })
    
    # Compute category overlaps
    overlaps = compute_category_overlaps(analysis_results['operations'])
    
    # Keep only essential overlap data for plots (remove triplets/quads not used by visualization)
    essential_overlaps = {
        'all_categories': overlaps['all_categories'],
        'co_occurrence_pairs': overlaps['co_occurrence_pairs'],
        'category_counts_per_operation': overlaps['category_counts_per_operation'],
    }
    
    # Add overlap data to export
    full_export = {
        'operations': export_data,
        'overlaps': essential_overlaps
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_export, f, indent=2, ensure_ascii=False)
    
    print(f"All codes saved to: {output_file}")


def main(output_dir: Path):
    """Main function to run the analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ANALYZING RUN_CODE OPERATIONS")
    print("="*60)
    
    # Analyze all runs
    analysis_results = analyze_all_runs(output_dir.parent)
    
    # Generate report
    report_file = output_dir / 'run_code_analysis_report.txt'
    generate_report(analysis_results, report_file)
    
    # Save all codes to JSON
    json_file = output_dir / 'run_code_operations.json'
    save_all_codes_json(analysis_results, json_file)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Total operations analyzed: {len(analysis_results['operations'])}")
    print(f"Report: {report_file.absolute()}")
    print(f"JSON data: {json_file.absolute()}")
    print("="*60)
