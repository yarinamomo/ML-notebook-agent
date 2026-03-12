"""
Find instances where "Submitted" action appears on early steps.
"""

import json
from pathlib import Path
from collections import defaultdict


def find_early_submissions(base_path: Path, max_step=5):
    """
    Find all instances where submission happens on early steps.
    
    Args:
        base_path: Base directory containing run folders
        max_step: Maximum step to consider as "early" (default: 5)
    
    Returns:
        Dictionary of findings
    """
    findings = defaultdict(list)
    all_submissions = []
    all_instance_statuses = []
    
    for run_num in [1, 2, 3]:
        run_dir = base_path / f'run_{run_num}'
        
        if not run_dir.exists():
            continue
        
        print(f"\nChecking {run_dir}...")
        
        # Find all summary files
        for summary_file in run_dir.glob('*_summary.json'):
            instance_id = summary_file.stem.replace('_summary', '')
            
            try:
                with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
                    summary = json.load(f)

                # Track final status for every instance summary (not only submissions)
                all_instance_statuses.append({
                    'run': run_num,
                    'instance': instance_id,
                    'status': summary.get('metadata', {}).get('status', 'N/A'),
                    'total_steps': summary.get('statistics', {}).get('total_steps', 'N/A')
                })
                
                # Check operations for submissions
                if 'operations' in summary:
                    for op in summary['operations']:
                        step = op.get('step')
                        action = op.get('action', '')
                        
                        # Check if this is a submission action
                        if 'submit' in action.lower() or action == 'Submitted':
                            submission_info = {
                                'run': run_num,
                                'instance': instance_id,
                                'step': step,
                                'action': action,
                                'success': op.get('success'),
                                'total_steps': summary.get('statistics', {}).get('total_steps', 'N/A'),
                                'status': summary.get('metadata', {}).get('status', 'N/A')
                            }
                            
                            all_submissions.append(submission_info)
                            
                            # Flag early submissions
                            if step is not None and step <= max_step:
                                findings[f'step_{step}'].append(submission_info)
            
            except Exception as e:
                print(f"  Error processing {summary_file.name}: {e}")
    
    return findings, all_submissions, all_instance_statuses

def save_to_file(findings, all_submissions, all_instance_statuses, output_file, max_step):
    """Save findings to a file."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EARLY SUBMISSION ANALYSIS - DETAILED LIST\n")
        f.write("="*80 + "\n\n")
        
        # Early submissions
        if findings:
            f.write(f"Found {sum(len(v) for v in findings.values())} early submissions (step <= {max_step}):\n\n")
            
            for step_key in sorted(findings.keys()):
                submissions = findings[step_key]
                step_num = step_key.split('_')[1]
                
                f.write(f"\n{'='*80}\n")
                f.write(f"STEP {step_num} SUBMISSIONS ({len(submissions)} instances)\n")
                f.write(f"{'='*80}\n")
                
                for sub in submissions:
                    f.write(f"\nRun {sub['run']} - {sub['instance']}\n")
                    f.write(f"  Action: {sub['action']}\n")
                    f.write(f"  Step: {sub['step']} / {sub['total_steps']} total\n")
                    f.write(f"  Status: {sub['status']}\n")
                    f.write(f"  Success: {sub['success']}\n")
        else:
            f.write(f"No early submissions found (step <= {max_step}).\n")
        # Status summary for all instance summaries
        f.write(f"\n\n{'='*80}\n")
        f.write("ALL INSTANCES BY FINAL STATUS\n")
        f.write(f"{'='*80}\n\n")

        final_status_counts = defaultdict(int)
        for item in all_instance_statuses:
            status_value = item.get('status', 'N/A')
            if status_value is None:
                status_value = 'N/A'
            final_status_counts[str(status_value)] += 1

        if final_status_counts:
            for status in sorted(final_status_counts.keys()):
                f.write(f"{status}: {final_status_counts[status]}\n")
        else:
            f.write("No instance summaries found.\n")

        # Detailed final status list for all instances
        f.write(f"\n\n{'='*80}\n")
        f.write("ALL INSTANCES (sorted by run, instance)\n")
        f.write(f"{'='*80}\n\n")

        for item in sorted(all_instance_statuses, key=lambda x: (x['run'], x['instance'])):
            f.write(f"Run {item['run']} | {item['instance']:<30s} | {item['status']}\n")
        
        # All submissions
        f.write(f"\n\n{'='*80}\n")
        f.write("ALL SUBMISSIONS (sorted by step)\n")
        f.write(f"{'='*80}\n\n")
        
        for sub in sorted(all_submissions, key=lambda x: (x['step'], x['run'], x['instance'])):
            f.write(f"Step {sub['step']:2d} | Run {sub['run']} | {sub['instance']:<30s} | {sub['status']}\n")
    
    print(f"\nDetailed list saved to: {output_file}")


def main(base_dir: Path, max_step=2):
    
    findings, all_submissions, all_instance_statuses = find_early_submissions(base_dir, max_step=max_step)
    
    # Create output directory if it doesn't exist
    output_dir = Path(base_dir) / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'early_submissions_report.txt'
    save_to_file(findings, all_submissions, all_instance_statuses, output_file, max_step=max_step)
