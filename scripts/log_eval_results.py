#!/usr/bin/env python3
"""Re-upload local evaluation results to MLflow.

Supports offline re-upload with fingerprint-based deduplication.
Use when MLflow was unavailable during evaluation or for re-uploading
corrected results.

Usage:
    python scripts/log_eval_results.py --results-dir data/evaluation/results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def get_existing_fingerprints(experiment_name: str) -> set[str]:
    """Query MLflow for existing run fingerprints to avoid duplicates."""
    try:
        import mlflow

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return set()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            max_results=5000,
        )

        fingerprints = set()
        if "params.fingerprint" in runs.columns:
            fingerprints = set(runs["params.fingerprint"].dropna().unique())

        return fingerprints
    except Exception as exc:
        console.print(f"[yellow]Could not query existing fingerprints: {exc}[/yellow]")
        return set()


def upload_result(result_path: Path, experiment_name: str, dry_run: bool = False) -> tuple[bool, str]:
    """Upload a single result file to MLflow.

    Returns (success, message).
    """
    try:
        data = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"Cannot read {result_path}: {exc}"

    track = data.get("track", data.get("params", {}).get("track", "unknown"))
    variant = data.get("variant", data.get("params", {}).get("variant", "all"))
    fingerprint = data.get("fingerprint", "")
    existing_run_id = data.get("mlflow_run_id", "")

    if not fingerprint:
        return False, f"No fingerprint in {result_path.name}"

    if dry_run:
        return True, f"Would upload: {result_path.name} (track={track}, variant={variant})"

    try:
        import mlflow

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{track}_{variant}_reupload") as run:
            # Log parameters
            mlflow.log_param("track", track)
            mlflow.log_param("variant", variant)
            mlflow.log_param("fingerprint", fingerprint)
            mlflow.log_param("config_hash", data.get("config_hash", ""))
            mlflow.log_param("model_hash", data.get("model_hash", ""))
            mlflow.log_param("data_revision", data.get("data_revision", ""))
            mlflow.log_param("original_timestamp", data.get("timestamp", ""))
            mlflow.log_param("reupload", "true")

            if existing_run_id:
                mlflow.log_param("original_run_id", existing_run_id)

            # Log additional params
            params = data.get("params", {})
            for key, value in params.items():
                try:
                    mlflow.log_param(key, value)
                except Exception:
                    pass

            # Log metrics
            metrics = data.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    try:
                        mlflow.log_metric(key, value)
                    except Exception:
                        pass

            # Log the result file itself as an artifact
            mlflow.log_artifact(str(result_path))

            # Log per-task results if referenced
            task_results_path = data.get("task_results_path", "")
            if task_results_path and Path(task_results_path).exists():
                mlflow.log_artifact(task_results_path)

            return True, f"Uploaded as run {run.info.run_id}"

    except ImportError:
        return False, "mlflow not installed"
    except Exception as exc:
        return False, f"MLflow error: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-upload local evaluation results to MLflow with deduplication.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing evaluation result JSON files",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name (default: from env MLFLOW_EXPERIMENT_EVAL)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even if fingerprint already exists in MLflow",
    )
    parser.add_argument(
        "--filter-track",
        type=str,
        default=None,
        help="Only upload results for the specified track",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        from rhoai_model_training_lab.config import load_env

        load_env()
    except Exception:
        pass

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    experiment_name = args.experiment or os.environ.get(
        "MLFLOW_EXPERIMENT_EVAL", "rhoai-model-training-lab-evaluation"
    )

    if not tracking_uri and not args.dry_run:
        console.print("[red]MLFLOW_TRACKING_URI not set. Use --dry-run to preview.[/red]")
        sys.exit(1)

    if tracking_uri and not args.dry_run:
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
        except ImportError:
            console.print("[red]mlflow not installed. Install with: pip install -e '.[eval]'[/red]")
            sys.exit(1)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        console.print(f"[red]Results directory not found: {results_dir}[/red]")
        sys.exit(1)

    # Find result files
    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        console.print(f"[yellow]No JSON files found in {results_dir}[/yellow]")
        sys.exit(0)

    console.print(f"[bold]═══ MLflow Result Upload ═══[/bold]")
    console.print(f"  Results dir: {results_dir}")
    console.print(f"  Files found: {len(result_files)}")
    console.print(f"  Experiment: {experiment_name}")
    console.print(f"  Dry run: {args.dry_run}")
    console.print(f"  Force: {args.force}")
    console.print()

    # Get existing fingerprints for deduplication
    existing_fingerprints: set[str] = set()
    if not args.force and not args.dry_run:
        console.print("Checking existing runs for deduplication...")
        existing_fingerprints = get_existing_fingerprints(experiment_name)
        console.print(f"  Found {len(existing_fingerprints)} existing fingerprints")
        console.print()

    # Process each file
    uploaded = 0
    skipped = 0
    failed = 0
    results_table = Table()
    results_table.add_column("File", style="bold")
    results_table.add_column("Track")
    results_table.add_column("Status")
    results_table.add_column("Details")

    for result_file in result_files:
        try:
            data = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError):
            results_table.add_row(result_file.name, "?", "❌ SKIP", "Cannot parse JSON")
            failed += 1
            continue

        track = data.get("track", data.get("params", {}).get("track", "unknown"))
        fingerprint = data.get("fingerprint", "")

        # Filter by track
        if args.filter_track and track != args.filter_track:
            continue

        # Deduplication check
        if fingerprint and fingerprint in existing_fingerprints and not args.force:
            results_table.add_row(result_file.name, track, "⏭️  SKIP", "Fingerprint exists")
            skipped += 1
            continue

        # Upload
        success, message = upload_result(result_file, experiment_name, dry_run=args.dry_run)
        if success:
            status_str = "✅ OK" if not args.dry_run else "🔍 DRY"
            results_table.add_row(result_file.name, track, status_str, message)
            uploaded += 1
        else:
            results_table.add_row(result_file.name, track, "❌ FAIL", message)
            failed += 1

    console.print(results_table)

    # Summary
    console.print()
    console.print("[bold]═══ Upload Summary ═══[/bold]")
    console.print(f"  Uploaded: {uploaded}")
    console.print(f"  Skipped (duplicate): {skipped}")
    console.print(f"  Failed: {failed}")

    if failed > 0:
        console.print(f"\n[yellow]⚠️  {failed} file(s) failed. Check errors above.[/yellow]")
    elif uploaded > 0:
        action = "would be uploaded" if args.dry_run else "uploaded"
        console.print(f"\n[green]✅  {uploaded} result(s) {action} to MLflow[/green]")
    else:
        console.print("\n[dim]No new results to upload.[/dim]")


if __name__ == "__main__":
    main()
