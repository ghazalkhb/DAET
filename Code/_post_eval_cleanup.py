# ==============================================================================
# _post_eval_cleanup.py
# Post-evaluation cleanup: removes obsolete result files, old transformer
# scripts, and old single-dataset pipeline scripts that are now superseded
# by run_full_evaluation.py.
#
# Run AFTER run_full_evaluation.py completes.
# Usage: python Code/_post_eval_cleanup.py
# ==============================================================================

import os, shutil, json, glob

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CODE    = os.path.join(ROOT, "Code")

# ==============================================================================
# 1. Results to remove (from old single-dataset pipeline runs)
# ==============================================================================

OLD_RESULT_FILES = [
    # Old two-way evaluation outputs
    "evaluation_all.csv",
    "evaluation_summary.json",
    "evaluation_arima.csv",
    "evaluation_last.csv",
    "evaluation_sma.csv",
    "evaluation_ewma.csv",
    "mozilla_eval_signatures.csv",
    # Old three-way (single-dataset)
    "three_way_summary.json",
    "three_way_detail.csv",
    "three_way_transformer_seeds.csv",
    # Smoke test artifacts
    "_smoke_out.txt", "_smoke2.txt", "_smoke3.txt",
    "_out.txt", "_err.txt", "_so.txt", "_se.txt",
    "_eval_out.txt", "_eval_err.txt", "_smokeout.txt", "_smokeerr.txt",
    "_smoke_check.txt",
    # Old debug files
    "_pass1.txt", "_pass1err.txt",
]

OLD_RESULT_DIRS = [
    "vectors",      # will be regenerated if needed; loads from timeseries-data now
]

# ==============================================================================
# 2. Old Code scripts superseded by run_full_evaluation.py
# ==============================================================================

OLD_CODE_FILES = [
    # Old separate step scripts (superseded by unified pipeline)
    "01_prepare_eval_signatures.py",
    "02_convert_to_vectors.py",
    "03_arima_detector.py",
    "04_last_detector.py",
    "05_sma_detector.py",
    "06_evaluate_arima.py",
    "07_evaluate_last.py",
    "08_evaluate_sma.py",
    "09_ewma_detector.py",
    # Old two-way pipeline
    "run_pipeline.py",
    # Old transformer scripts (replaced by unified Transformer in run_full_evaluation.py)
    "run_transformer.py",
    "run_transformer_v2.py",
    "run_transformer_v3.py",
    "run_transformer_fix_cdef.py",
    "run_transformer_complete.py",
    # Old three-way (single-dataset only)
    "12_three_way_eval.py",
    "13_transformer_three_way_eval.py",
    # Old master runners / debug scripts
    "_master_runner.py",
    "_audit.py",
    "_check_ablation.py",
    "_check_csv.py",
    "_check_inventory.py",
    "_check_status.py",
    "_fix_eval_all.py",
    "_fix_results.py",
    "_inspect_results.py",
    "_print_ablation.py",
    "_print_stats.py",
    "_speed_test.py",
    "_temp_check.py",
    "_temp_diag.py",
    "_verify_final.py",
    "_verify_fixes.py",
    "_write_report_gen.py",
    "_smoke_test.py",
]

# ==============================================================================
# 3. Old result text/log files in results/
# ==============================================================================

OLD_RESULT_PATTERNS = [
    "_ablation*.txt",
    "_audit*.txt",
    "_col*.txt",
    "_comp*.txt",
    "_err*.txt",
    "_eval*.txt",
    "_extract*.txt",
    "_extracted*.txt",
    "_fix*.txt",
    "_full*.py",
    "_latex*.txt",
    "_log*.txt",
    "_now.txt",
    "_pass*.txt",
    "_pdf*.txt",
    "_proc*.txt",
    "_procA*.txt",
]


def remove_file(path, dry_run=False):
    if os.path.isfile(path):
        if dry_run:
            print(f"  [DRY] Would remove: {path}")
        else:
            os.remove(path)
            print(f"  Removed: {os.path.basename(path)}")
    else:
        pass  # silently skip missing files


def remove_dir(path, dry_run=False):
    if os.path.isdir(path):
        if dry_run:
            print(f"  [DRY] Would remove dir: {path}")
        else:
            shutil.rmtree(path)
            print(f"  Removed dir: {os.path.basename(path)}")


def cleanup(dry_run=False):
    tag = "[DRY RUN]" if dry_run else ""
    print(f"\n{'='*60}")
    print(f" Post-Evaluation Cleanup {tag}")
    print(f"{'='*60}")

    # ── Results files ────────────────────────────────────────────────
    print("\n-- Old result files --")
    for f in OLD_RESULT_FILES:
        remove_file(os.path.join(RESULTS, f), dry_run)

    print("\n-- Old result directories --")
    for d in OLD_RESULT_DIRS:
        remove_dir(os.path.join(RESULTS, d), dry_run)

    print("\n-- Old result log patterns --")
    for pat in OLD_RESULT_PATTERNS:
        for path in glob.glob(os.path.join(RESULTS, pat)):
            remove_file(path, dry_run)

    # ── Code files ───────────────────────────────────────────────────
    print("\n-- Old Code scripts --")
    for f in OLD_CODE_FILES:
        remove_file(os.path.join(CODE, f), dry_run)

    # ── Old Data folders (empty after zip extraction leftover) ───────
    print("\n-- Old empty Data folders --")
    for name in ["firefox-android", "mozilla-central", "mozilla-release"]:
        p = os.path.join(ROOT, "Data", name)
        if os.path.isdir(p):
            items = os.listdir(p)
            if not items:
                remove_dir(p, dry_run)
            else:
                print(f"  Skipping non-empty: Data/{name}/ ({len(items)} items)")

    # ── Verify core results exist ────────────────────────────────────
    print("\n-- Verifying new results --")
    for ds in ["firefox-android", "mozilla-beta"]:
        detail = os.path.join(RESULTS, ds, "detail.csv")
        summ   = os.path.join(RESULTS, ds, "summary.json")
        ok_d   = os.path.isfile(detail)
        ok_s   = os.path.isfile(summ)
        print(f"  {ds}/detail.csv: {'OK' if ok_d else 'MISSING'}")
        print(f"  {ds}/summary.json: {'OK' if ok_s else 'MISSING'}")

    combined = os.path.join(RESULTS, "combined_summary.json")
    print(f"  combined_summary.json: {'OK' if os.path.isfile(combined) else 'MISSING'}")

    print("\nCleanup complete.\n")


if __name__ == "__main__":
    import sys
    dry = "--dry" in sys.argv
    cleanup(dry_run=dry)
