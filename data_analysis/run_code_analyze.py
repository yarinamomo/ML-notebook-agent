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
        categories.append('print_output')

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
        categories.append('runinfo_structural')

    # type semantics (type, dtype, shcema-level properties)
    if any(pattern in code for pattern in ['type(', '.dtypes', 'isinstance(', 'dtype']):
        categories.append('runinfo_type')

    # value semantics (concrete values: value range, presence of NaNs, number of classes, example values)
    if (
        any(pattern in code for pattern in ['.head(', '.tail(', '.sample(', '.info(', '.describe(', '.columns', '.index', '.values', '.keys', '.listdir(', '.class_names', '.iloc', '.loc']) 
        or ('in' in code and 'columns' in code)
        or any(re.search(pattern, code) for pattern in [r'\.read\(\d+\)', r'\[\s*-?\d*\s*:\s*-?\d*\s*\]'])
    ):
        categories.append('runinfo_value')
    
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
        
        # Analyze code lengths
        code_lengths = [len(op['code']) for op in operations]
        avg_length = sum(code_lengths) / len(code_lengths) if code_lengths else 0
        f.write(f"Average code length: {avg_length:.0f} characters\n")
        f.write(f"Shortest code: {min(code_lengths)} characters\n")
        f.write(f"Longest code: {max(code_lengths)} characters\n\n")
        
        # Multi-category operations
        multi_cat_ops = [op for op in operations if len(op['categories']) > 1]
        f.write(f"Operations with multiple categories: {len(multi_cat_ops)} ({len(multi_cat_ops)/len(operations)*100:.1f}%)\n")
    
    print(f"\nReport saved to: {output_file}")


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
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"All codes saved to: {output_file}")


def main(output_dir: Path):
    """Main function to run the analysis."""
    output_dir.mkdir(exist_ok=True)
    
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


if __name__ == '__main__':
    main(Path('trajectories_monday/glm-4.7-355b/analysis'))
