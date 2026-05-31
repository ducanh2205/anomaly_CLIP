"""
Master submission pipeline: inference → convert → validate → compress.

Usage:
  # Step 1: Run batch inference (already done)
  python batch_inference_private.py --output_dir ./results
  
  # Step 2: Convert to submission format (already done)
  python convert_to_submission_format.py --results_dir ./results --output_dir ./submission_private
  
  # Step 3: Validate with official tool
  python ../MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py ./submission_private
  
  # Step 4: Upload to benchmark.mvtec.com
"""

import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).parent
    mvtec_utils_dir = script_dir.parent / "MVTecAD2_public_code_utils"
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    MVTec AD 2 Submission Pipeline                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check if inference is done
    results_dir = script_dir / "results"
    if not results_dir.exists():
        print("[ERROR] results/ not found. Run batch_inference_private.py first.")
        sys.exit(1)
    
    # Check completeness of results
    print("[Step 1] Checking inference results...")
    import csv
    from collections import defaultdict
    
    obj_splits = defaultdict(set)
    for obj_dir in results_dir.iterdir():
        if not obj_dir.is_dir():
            continue
        obj = obj_dir.name
        csv_file = obj_dir / "results_summary.csv"
        if csv_file.exists():
            with open(csv_file) as f:
                count = sum(1 for _ in csv.reader(f)) - 1  # -1 for header
                print(f"  {obj}: {count} images")
            obj_splits[obj].add('all')
    
    print(f"\n  Objects found: {len(obj_splits)}")
    for obj in sorted(obj_splits.keys()):
        splits = sorted(obj_splits[obj])
        print(f"    {obj}: {', '.join(splits)}")
    
    # Step 2: Convert (already done separately)
    submission_dir = script_dir / "submission_private"
    if not submission_dir.exists():
        print(f"[ERROR] submission_private/ not found. Run convert_to_submission_format.py first.")
        sys.exit(1)
    print(f"[Step 2] Submission directory found: {submission_dir}")
    
    # Step 3: Validate
    print(f"\n[Step 3] Validating submission...")
    submission_dir = script_dir / "submission_private"
    try:
        result = subprocess.run(
            [sys.executable, 
             str(mvtec_utils_dir / "check_and_prepare_data_for_upload.py"),
             str(submission_dir)],
            cwd=str(script_dir),
            capture_output=False
        )
        if result.returncode != 0:
            print(f"[ERROR] Validation failed!")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Validation script error: {e}")
        sys.exit(1)
    
    # Done
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                            ✓ Ready to Submit!                             ║
║                                                                           ║
║  1. Compressed submission: {submission_dir}.zip                    ║
║  2. Upload to: https://www.benchmark.mvtec.com                            ║
║  3. Check evaluation results in leaderboard                               ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    main()
