#!/usr/bin/env python3
"""Main evaluation runner for the RHOAI Model Training Lab.

Supports three evaluation tracks:
  - prepared_diagnostics: Offline holdout QA/policy/tool diagnostics
  - tau_episodes: End-to-end τ-Knowledge official evaluation
  - retention: Capability preservation benchmarks (ARC-Challenge, etc.)

Usage:
    python scripts/run_eval.py --track prepared_diagnostics --config configs/eval.yaml
    python scripts/run_eval.py --track tau_episodes --config configs/eval.yaml --limit 5 --trials 1
    python scripts/run_eval.py --track retention --config configs/eval.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def compute_config_hash(config: dict) -> str:
    """Compute a deterministic hash of the evaluation configuration."""
    config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_run_fingerprint(
    track: str,
    variant: str,
    model_hash: str,
    data_revision: str,
    config_hash: str,
) -> str:
    """Compute a unique fingerprint for deduplication."""
    parts = f"{track}:{variant}:{model_hash}:{data_revision}:{config_hash}"
    return hashlib.sha256(parts.encode()).hexdigest()[:24]


def save_results_local(results: dict, output_dir: Path, track: str, variant: str) -> Path:
    """Save evaluation results to local filesystem first (before MLflow)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{track}_{variant}_{timestamp}.json" if variant else f"{track}_{timestamp}.json"
    result_path = output_dir / filename
    result_path.write_text(json.dumps(results, indent=2, default=str))
    return result_path


def log_to_mlflow(
    results: dict,
    track: str,
    variant: str,
    eval_config: dict,
    no_mlflow: bool = False,
) -> str | None:
    """Log evaluation results to MLflow. Returns run_id or None."""
    if no_mlflow:
        console.print("[dim]MLflow logging disabled (--no-mlflow)[/dim]")
        return None

    mlflow_cfg = eval_config.get("mlflow", {})
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get("tracking_uri", "")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_EVAL") or mlflow_cfg.get("experiment_name", "rhoai-model-training-lab-evaluation")

    if not tracking_uri:
        console.print("[yellow]MLFLOW_TRACKING_URI not set — skipping MLflow logging[/yellow]")
        return None

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{track}_{variant}") as run:
            # Log parameters
            mlflow.log_param("track", track)
            mlflow.log_param("variant", variant)
            mlflow.log_param("config_hash", results.get("config_hash", ""))
            mlflow.log_param("fingerprint", results.get("fingerprint", ""))
            mlflow.log_param("model_hash", results.get("model_hash", ""))
            mlflow.log_param("data_revision", results.get("data_revision", ""))

            params = results.get("params", {})
            for key, value in params.items():
                try:
                    mlflow.log_param(key, value)
                except Exception:
                    pass

            # Log metrics
            metrics = results.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    try:
                        mlflow.log_metric(key, value)
                    except Exception:
                        pass

            # Log artifacts
            for artifact_key in ("task_results_path", "local_result_path"):
                artifact_path = results.get(artifact_key, "")
                if artifact_path and Path(artifact_path).exists():
                    try:
                        mlflow.log_artifact(artifact_path)
                    except Exception:
                        pass

            return run.info.run_id

    except ImportError:
        console.print("[yellow]mlflow not installed — skipping MLflow logging[/yellow]")
        return None
    except Exception as exc:
        console.print(f"[red]MLflow logging failed: {exc}[/red]")
        console.print("[yellow]Results were saved locally. Use log_eval_results.py to re-upload.[/yellow]")
        return None


def run_prepared_diagnostics(eval_config: dict, track_config: dict, args: argparse.Namespace) -> dict:
    """Run prepared offline diagnostics evaluation."""
    holdout_path = Path(track_config.get("holdout_path", ""))
    matched_context = track_config.get("matched_context", True)
    scoring_cfg = track_config.get("scoring", {})

    if not holdout_path.exists():
        console.print(f"[red]Holdout file not found: {holdout_path}[/red]")
        return {"status": "failed", "error": f"Holdout not found: {holdout_path}"}

    console.print(f"[bold]Loading holdout data from: {holdout_path}[/bold]")

    # Load holdout samples
    samples: list[dict] = []
    with open(holdout_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if args.limit:
        samples = samples[: args.limit]

    console.print(f"  Loaded {len(samples)} holdout samples")

    # Filter by variant types if specified
    requested_types = track_config.get("types", [])
    if requested_types:
        samples = [s for s in samples if s.get("sample_type") in requested_types]
        console.print(f"  Filtered to {len(samples)} samples (types: {requested_types})")

    # Run evaluation using the evaluation module
    try:
        from rhoai_model_training_lab.evaluation import diagnostics

        results = diagnostics.run(
            samples=samples,
            scoring=scoring_cfg,
            matched_context=matched_context,
            variant=args.variant or "all",
        )
    except (ImportError, AttributeError):
        console.print("[yellow]Evaluation module not fully implemented — running basic checks[/yellow]")
        results = _basic_diagnostic_eval(samples, scoring_cfg)

    return results


def _basic_diagnostic_eval(samples: list[dict], scoring_cfg: dict) -> dict:
    """Basic diagnostic evaluation fallback."""
    total = len(samples)
    by_type: dict[str, int] = {}
    for s in samples:
        stype = s.get("sample_type", "unknown")
        by_type[stype] = by_type.get(stype, 0) + 1

    return {
        "status": "completed",
        "track": "prepared_diagnostics",
        "total_samples": total,
        "by_type": by_type,
        "metrics": {
            "total_evaluated": total,
            "answer_accuracy": None,
            "tool_name_accuracy": None,
            "tool_args_accuracy": None,
            "grounding_score": None,
        },
        "note": "τ-Knowledge-derived lab diagnostics (not official τ pass rates)",
    }


def run_tau_episodes(eval_config: dict, track_config: dict, args: argparse.Namespace) -> dict:
    """Run end-to-end τ-Knowledge episode evaluation."""
    tau_version = track_config.get("tau_version", "")
    domain = track_config.get("domain", "banking_knowledge")
    variants = track_config.get("variants", [])
    ablations = track_config.get("ablations", [])
    grading_cfg = track_config.get("grading", {})
    budget_cfg = track_config.get("budgets", {})

    trials = args.trials if args.trials is not None else track_config.get("trials", 3)
    limit = args.limit if args.limit is not None else track_config.get("limit")

    # Filter variants
    all_variants = variants + ablations
    if args.variant:
        all_variants = [v for v in all_variants if v.get("name") == args.variant]
        if not all_variants:
            console.print(f"[red]Variant '{args.variant}' not found[/red]")
            available = [v.get("name") for v in variants + ablations]
            console.print(f"[dim]Available: {available}[/dim]")
            return {"status": "failed", "error": f"Unknown variant: {args.variant}"}

    console.print(f"[bold]τ-Knowledge Episode Evaluation[/bold]")
    console.print(f"  Domain: {domain}")
    console.print(f"  τ version: {tau_version}")
    console.print(f"  Trials: {trials}")
    console.print(f"  Limit: {limit or 'all'}")
    console.print(f"  Variants: {[v.get('name') for v in all_variants]}")

    # Run evaluation for each variant
    all_results: list[dict] = []

    for variant_cfg in all_variants:
        variant_name = variant_cfg.get("name", "unknown")
        model_endpoint_key = variant_cfg.get("model_endpoint", "base")
        knowledge_access = variant_cfg.get("knowledge_access", "no_knowledge")
        mode = variant_cfg.get("mode", "agent_rag")

        console.print(f"\n[bold]Running variant: {variant_name}[/bold]")
        console.print(f"  Model: {model_endpoint_key}, Knowledge: {knowledge_access}, Mode: {mode}")

        try:
            from rhoai_model_training_lab.evaluation import tau_runner

            variant_result = tau_runner.run_episodes(
                domain=domain,
                model_endpoint=model_endpoint_key,
                knowledge_access=knowledge_access,
                mode=mode,
                trials=trials,
                limit=limit,
                budgets=budget_cfg,
                grading=grading_cfg,
            )
        except (ImportError, AttributeError):
            console.print(f"  [yellow]τ runner not fully implemented — recording placeholder[/yellow]")
            variant_result = {
                "variant": variant_name,
                "status": "not_run",
                "model_endpoint": model_endpoint_key,
                "knowledge_access": knowledge_access,
                "mode": mode,
                "trials": trials,
                "limit": limit,
                "metrics": {},
                "note": "τ runner not yet available — install and configure τ-bench",
            }

        variant_result["variant"] = variant_name
        all_results.append(variant_result)

    # Aggregate
    return {
        "status": "completed" if all_results else "failed",
        "track": "tau_episodes",
        "domain": domain,
        "tau_version": tau_version,
        "trials": trials,
        "limit": limit,
        "variants": all_results,
        "metrics": {},
    }


def run_retention(eval_config: dict, track_config: dict, args: argparse.Namespace) -> dict:
    """Run capability retention evaluation."""
    benchmarks = track_config.get("benchmarks", [])

    console.print("[bold]Retention Evaluation[/bold]")
    console.print(f"  Benchmarks: {[b.get('name') for b in benchmarks]}")
    console.print(f"  KB/RAG disabled: {track_config.get('disable_kb_rag', True)}")

    all_results: list[dict] = []

    for bench_cfg in benchmarks:
        bench_name = bench_cfg.get("name", "unknown")
        bench_type = bench_cfg.get("type", "generated_answer")
        sample_size = bench_cfg.get("sample_size", 100)
        seed = bench_cfg.get("seed", 42)

        if args.limit:
            sample_size = min(sample_size, args.limit)

        console.print(f"\n[bold]Benchmark: {bench_name}[/bold]")
        console.print(f"  Type: {bench_type}, Samples: {sample_size}")

        # Run for each model variant
        for variant_name in ("base", "lora", "osft"):
            console.print(f"  Running: {variant_name}...")

            try:
                from rhoai_model_training_lab.evaluation import retention

                bench_result = retention.evaluate(
                    benchmark=bench_name,
                    variant=variant_name,
                    sample_size=sample_size,
                    seed=seed,
                    bench_type=bench_type,
                )
            except (ImportError, AttributeError):
                bench_result = {
                    "benchmark": bench_name,
                    "variant": variant_name,
                    "status": "not_run",
                    "sample_size": sample_size,
                    "accuracy": None,
                    "base_accuracy": None,
                    "retention_delta_pp": None,
                    "note": "Retention evaluation not yet implemented",
                }

            all_results.append(bench_result)

    return {
        "status": "completed" if all_results else "failed",
        "track": "retention",
        "benchmarks": all_results,
        "metrics": {},
    }


TRACK_RUNNERS = {
    "prepared_diagnostics": run_prepared_diagnostics,
    "tau_episodes": run_tau_episodes,
    "retention": run_retention,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RHOAI Model Training Lab — Evaluation Runner",
    )
    parser.add_argument(
        "--track",
        type=str,
        required=True,
        choices=["prepared_diagnostics", "tau_episodes", "retention"],
        help="Evaluation track to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Evaluation configuration file (default: configs/eval.yaml)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks/samples",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of trials per task (tau_episodes only)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Run only the specified variant",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging (offline mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for results",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        from rhoai_model_training_lab.config import load_env, load_yaml_config

        load_env()
        eval_config = load_yaml_config(args.config)
    except Exception as exc:
        console.print(f"[red]Failed to load config: {exc}[/red]")
        sys.exit(1)

    general_cfg = eval_config.get("general", {})
    tracks_cfg = eval_config.get("tracks", {})

    no_mlflow = args.no_mlflow or general_cfg.get("no_mlflow", False)
    output_dir = Path(args.output_dir or general_cfg.get("output_dir", "data/evaluation/results"))

    track_config = tracks_cfg.get(args.track, {})
    if not track_config.get("enabled", True):
        console.print(f"[yellow]Track '{args.track}' is disabled in config[/yellow]")
        sys.exit(0)

    config_hash = compute_config_hash(track_config)
    variant_label = args.variant or "all"

    console.print(f"[bold]═══ Evaluation: {args.track} ═══[/bold]")
    console.print(f"  Config: {args.config}")
    console.print(f"  Config hash: {config_hash}")
    console.print(f"  Variant: {variant_label}")
    console.print()

    # Run the track
    runner = TRACK_RUNNERS[args.track]
    start_time = time.time()
    results = runner(eval_config, track_config, args)
    elapsed = time.time() - start_time

    # Add metadata
    fingerprint = compute_run_fingerprint(
        track=args.track,
        variant=variant_label,
        model_hash="",
        data_revision="",
        config_hash=config_hash,
    )

    results["fingerprint"] = fingerprint
    results["config_hash"] = config_hash
    results["timestamp"] = datetime.utcnow().isoformat()
    results["elapsed_seconds"] = round(elapsed, 2)
    results["model_hash"] = ""
    results["data_revision"] = ""
    results["params"] = {
        "track": args.track,
        "variant": variant_label,
        "limit": args.limit,
        "trials": args.trials,
    }

    # Save local results first
    console.print()
    result_path = save_results_local(results, output_dir, args.track, variant_label)
    results["local_result_path"] = str(result_path)
    console.print(f"[green]Results saved: {result_path}[/green]")

    # Log to MLflow
    run_id = log_to_mlflow(results, args.track, variant_label, eval_config, no_mlflow)
    if run_id:
        console.print(f"[green]MLflow run: {run_id}[/green]")
        results["mlflow_run_id"] = run_id
        # Update local file with MLflow run ID
        result_path.write_text(json.dumps(results, indent=2, default=str))

    # Print summary
    console.print()
    console.print(f"[bold]═══ Evaluation Summary ═══[/bold]")

    summary_table = Table()
    summary_table.add_column("Field", style="bold")
    summary_table.add_column("Value")
    summary_table.add_row("Track", args.track)
    summary_table.add_row("Variant", variant_label)
    summary_table.add_row("Status", results.get("status", "unknown"))
    summary_table.add_row("Elapsed", f"{elapsed:.1f}s")
    summary_table.add_row("Fingerprint", fingerprint)
    summary_table.add_row("MLflow", run_id or "disabled/unavailable")
    summary_table.add_row("Results", str(result_path))
    console.print(summary_table)

    # Print metrics if available
    metrics = results.get("metrics", {})
    if metrics:
        console.print()
        metrics_table = Table(title="Metrics")
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", justify="right")
        for key, value in metrics.items():
            if value is not None:
                metrics_table.add_row(key, f"{value}")
        console.print(metrics_table)

    status = results.get("status", "failed")
    if status in ("completed", "not_run"):
        console.print(f"\n[green]✅  Evaluation complete: {args.track}[/green]")
    else:
        console.print(f"\n[red]❌  Evaluation failed: {args.track}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
