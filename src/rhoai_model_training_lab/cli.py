"""CLI entry point for the RHOAI Model Training Lab.

Provides commands for preflight checks, training, evaluation, result
re-upload, and backend serving.  Registered as ``rhoai-lab`` via
``pyproject.toml`` entry-point.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import click

from rhoai_model_training_lab.config import (
    PROJECT_ROOT,
    load_env,
    load_eval_config,
    load_training_config,
    load_yaml_config,
)

logger = logging.getLogger("rhoai_model_training_lab")


# ---------------------------------------------------------------------------
# Rich console helper
# ---------------------------------------------------------------------------

def _get_console() -> Any:
    """Return a rich Console, or a lightweight fallback."""
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        pass

    class _FallbackConsole:
        """Minimal shim used when rich is not installed."""

        @staticmethod
        def print(*args: Any, **kwargs: Any) -> None:
            text = " ".join(str(a) for a in args)
            click.echo(text)

        rule = print
        log = print

    return _FallbackConsole()


console = _get_console()


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logging with level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------

@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """RHOAI Model Training Lab — τ-Knowledge banking model training."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)
    load_env()


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

@main.command()
@click.pass_context
def preflight(ctx: click.Context) -> None:
    """Run preflight checks: GPU, environment, PVC, MLflow, model profile."""
    console.rule("[bold blue]Preflight Checks[/bold blue]")
    checks_passed = 0
    checks_failed = 0
    checks_warned = 0

    # 1. GPU check
    console.print("\n[bold]1. GPU Check[/bold]")
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            console.print(f"  ✅ GPU available: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            checks_passed += 1
        else:
            console.print("  ⚠️  No CUDA GPU detected. Training requires a GPU.")
            checks_warned += 1
    except ImportError:
        console.print("  ⚠️  PyTorch not installed. Install with: pip install torch")
        checks_warned += 1

    # 2. Environment variables
    console.print("\n[bold]2. Environment Variables[/bold]")
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        console.print(f"  ✅ .env file found: {env_path}")
        checks_passed += 1
    else:
        console.print("  ⚠️  No .env file. Copy .env.example to .env and configure.")
        checks_warned += 1

    required_training_vars = ["BASE_MODEL_ID"]
    optional_vars = [
        "MLFLOW_TRACKING_URI",
        "S3_ENDPOINT",
        "BASE_SERVING_ENDPOINT",
    ]
    for var in required_training_vars:
        val = os.environ.get(var, "")
        if val:
            console.print(f"  ✅ {var} = {val}")
            checks_passed += 1
        else:
            console.print(f"  ❌ {var} not set (required for training)")
            checks_failed += 1

    for var in optional_vars:
        val = os.environ.get(var, "")
        status = "✅" if val else "⚠️ "
        display = val[:60] + "..." if val and len(val) > 60 else (val or "(not set)")
        console.print(f"  {status} {var} = {display}")
        if val:
            checks_passed += 1
        else:
            checks_warned += 1

    # 3. PVC / data paths
    console.print("\n[bold]3. Data Paths[/bold]")
    data_dir = PROJECT_ROOT / "data"
    bundle_dir = data_dir / "prepared"
    if bundle_dir.exists():
        bundles = list(bundle_dir.iterdir())
        if bundles:
            console.print(f"  ✅ Prepared bundles found: {[b.name for b in bundles]}")
            checks_passed += 1
        else:
            console.print("  ⚠️  data/prepared/ exists but is empty")
            checks_warned += 1
    else:
        console.print("  ⚠️  No data/prepared/ directory. Fetch a bundle first.")
        checks_warned += 1

    # 4. MLflow
    console.print("\n[bold]4. MLflow[/bold]")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if mlflow_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_uri)
            experiments = mlflow.search_experiments(max_results=1)
            console.print(f"  ✅ MLflow reachable at {mlflow_uri}")
            checks_passed += 1
        except ImportError:
            console.print("  ⚠️  mlflow not installed. Install with: pip install mlflow")
            checks_warned += 1
        except Exception as exc:
            console.print(f"  ❌ MLflow connection failed: {exc}")
            checks_failed += 1
    else:
        console.print("  ⚠️  MLFLOW_TRACKING_URI not set; will use local ./mlruns")
        checks_warned += 1

    # 5. Model profile
    console.print("\n[bold]5. Model Profile[/bold]")
    model_id = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
    console.print(f"  📦 Model: {model_id}")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        console.print(f"  ✅ Tokenizer loaded: vocab_size={tok.vocab_size}")
        if hasattr(tok, "chat_template") and tok.chat_template:
            console.print("  ✅ Chat template available")
        checks_passed += 1
    except ImportError:
        console.print("  ⚠️  transformers not installed")
        checks_warned += 1
    except Exception as exc:
        console.print(f"  ⚠️  Could not load tokenizer: {exc}")
        checks_warned += 1

    # 6. Config files
    console.print("\n[bold]6. Configuration Files[/bold]")
    for cfg_name in ("configs/eval.yaml", "configs/lora.yaml", "configs/osft.yaml", "configs/rag.yaml"):
        cfg_path = PROJECT_ROOT / cfg_name
        if cfg_path.exists():
            console.print(f"  ✅ {cfg_name}")
            checks_passed += 1
        else:
            console.print(f"  ⚠️  {cfg_name} not found")
            checks_warned += 1

    # Summary
    console.rule("[bold blue]Summary[/bold blue]")
    console.print(f"  ✅ Passed: {checks_passed}")
    console.print(f"  ⚠️  Warnings: {checks_warned}")
    console.print(f"  ❌ Failed: {checks_failed}")

    if checks_failed > 0:
        console.print("\n[bold red]Preflight failed. Fix errors above before proceeding.[/bold red]")
        sys.exit(1)
    elif checks_warned > 0:
        console.print("\n[bold yellow]Preflight passed with warnings.[/bold yellow]")
    else:
        console.print("\n[bold green]All preflight checks passed![/bold green]")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--profile",
    type=click.Choice(["lora", "osft"]),
    required=True,
    help="Training profile: lora or osft.",
)
@click.option("--config", "config_path", default=None, help="Override config path.")
@click.option("--no-mlflow", is_flag=True, help="Disable MLflow logging.")
@click.option("--dry-run", is_flag=True, help="Validate config without training.")
@click.pass_context
def train(ctx: click.Context, profile: str, config_path: str | None, no_mlflow: bool, dry_run: bool) -> None:
    """Train a model with the specified profile (lora or osft).

    Delegates to the training module after validating the prepared dataset
    bundle.  Requires a GPU.
    """
    console.rule(f"[bold blue]Training: {profile.upper()}[/bold blue]")

    try:
        training_config = load_training_config(profile)
    except FileNotFoundError:
        console.print(f"[red]Configuration file not found: configs/{profile}.yaml[/red]")
        console.print("Create the file or copy from the example configuration.")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]Error loading config: {exc}[/red]")
        sys.exit(1)

    if config_path:
        override = load_yaml_config(config_path)
        training_config.update(override)

    console.print(f"  Profile: {profile}")
    console.print(f"  Model: {training_config.get('model_id', 'unknown')}")
    console.print(f"  MLflow: {'disabled' if no_mlflow else 'enabled'}")

    train_file = training_config.get("train_file", "")
    val_file = training_config.get("validation_file", "")
    if train_file:
        train_path = Path(train_file) if Path(train_file).is_absolute() else PROJECT_ROOT / train_file
        if not train_path.exists():
            console.print(f"[red]Training file not found: {train_path}[/red]")
            console.print("Run 'python scripts/fetch_prepared_dataset.py' to download the bundle.")
            sys.exit(1)
        console.print(f"  Train data: {train_path}")
    else:
        console.print("[yellow]Warning: train_file not configured[/yellow]")

    if val_file:
        val_path = Path(val_file) if Path(val_file).is_absolute() else PROJECT_ROOT / val_file
        if not val_path.exists():
            console.print(f"[red]Validation file not found: {val_path}[/red]")
            sys.exit(1)
        console.print(f"  Validation data: {val_path}")

    if dry_run:
        console.print("\n[yellow]Dry run — skipping training.[/yellow]")
        console.print("Configuration is valid.")
        return

    try:
        import torch

        if not torch.cuda.is_available():
            console.print("[red]No GPU available. Training requires a CUDA GPU.[/red]")
            sys.exit(1)
    except ImportError:
        console.print("[red]PyTorch not installed. Install with the appropriate profile.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Starting {profile.upper()} training...[/bold]")
    try:
        from rhoai_model_training_lab.training import run_training

        result = run_training(profile, training_config)

        console.print(f"\n[green]Training complete![/green]")
        console.print(f"  Method: {result.method}")
        console.print(f"  Steps: {result.total_steps}")
        console.print(f"  Final loss: {result.final_train_loss:.4f}")
        console.print(f"  Wall time: {result.wall_time_seconds:.1f}s")

        if not no_mlflow:
            from rhoai_model_training_lab.tracking import MLflowTracker

            tracker = MLflowTracker()
            run_id = tracker.log_training_run(result)
            if run_id:
                console.print(f"  MLflow run: {run_id}")
            else:
                console.print("[yellow]MLflow logging failed; result saved locally.[/yellow]")
        else:
            from rhoai_model_training_lab.tracking import LocalResultStore

            store = LocalResultStore()
            fp = store.save_training_result(result)
            console.print(f"  Saved locally (fp={fp})")

    except ImportError as exc:
        console.print(f"[red]Training module not available: {exc}[/red]")
        console.print(f"Install dependencies: pip install -e '.[{profile}]'")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Training failed")
        console.print(f"[red]Training failed: {exc}[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--track",
    type=click.Choice(["prepared_diagnostics", "tau_episodes", "retention", "all"]),
    default="all",
    help="Evaluation track to run.",
)
@click.option("--config", "config_path", default=None, help="Evaluation config path.")
@click.option("--limit", type=int, default=None, help="Max number of tasks to evaluate.")
@click.option("--trials", type=int, default=None, help="Number of trials per task.")
@click.option("--no-mlflow", is_flag=True, help="Disable MLflow logging.")
@click.option("--variant", default=None, help="Specific variant to evaluate.")
@click.option(
    "--endpoint",
    multiple=True,
    type=(str, str),
    help="Endpoint mapping as --endpoint NAME URL pairs.",
)
@click.pass_context
def evaluate(
    ctx: click.Context,
    track: str,
    config_path: str | None,
    limit: int | None,
    trials: int | None,
    no_mlflow: bool,
    variant: str | None,
    endpoint: tuple[tuple[str, str], ...],
) -> None:
    """Run evaluation tracks: diagnostics, τ episodes, retention, or all."""
    console.rule("[bold blue]Evaluation[/bold blue]")

    try:
        eval_config = load_eval_config(config_path)
    except FileNotFoundError:
        console.print("[red]Evaluation config not found. Create configs/eval.yaml.[/red]")
        sys.exit(1)

    endpoint_map: dict[str, str] = dict(endpoint)

    from rhoai_model_training_lab.schemas.evaluation import EvalRunConfig

    run_config = EvalRunConfig(
        track=track if track != "all" else "",
        variant=variant or "",
        limit=limit,
        trials=trials or eval_config.get("tracks", {}).get("tau_episodes", {}).get("trials", 3),
        no_mlflow=no_mlflow,
    )

    console.print(f"  Track: {track}")
    console.print(f"  Limit: {limit or 'all'}")
    console.print(f"  Trials: {run_config.trials}")
    console.print(f"  MLflow: {'disabled' if no_mlflow else 'enabled'}")
    if endpoint_map:
        for name, url in endpoint_map.items():
            console.print(f"  Endpoint {name}: {url}")

    from rhoai_model_training_lab.evaluation import EvalOrchestrator

    orchestrator = EvalOrchestrator(eval_config)

    try:
        if track == "all":
            all_results = orchestrator.run_all_tracks(
                run_config,
                limit=limit,
                trials=trials,
                endpoint_map=endpoint_map or None,
            )

            comparison_rows = orchestrator.build_comparison_table(all_results)

            if not no_mlflow:
                _log_all_results(all_results, comparison_rows, no_mlflow=False)
            else:
                _log_all_results(all_results, comparison_rows, no_mlflow=True)

            _print_summary(all_results, comparison_rows)

        else:
            results = orchestrator.run_track(
                track,
                run_config,
                limit=limit,
                trials=trials,
                endpoint_map=endpoint_map or None,
            )

            if not no_mlflow:
                for result in results:
                    _log_single_result(result, no_mlflow=False)
            else:
                for result in results:
                    _log_single_result(result, no_mlflow=True)

            for result in results:
                _print_result(result)

    except Exception as exc:
        logger.exception("Evaluation failed")
        console.print(f"[red]Evaluation failed: {exc}[/red]")
        sys.exit(1)


def _log_single_result(result: Any, *, no_mlflow: bool) -> None:
    """Log a single EvalResult to MLflow or local store."""
    from rhoai_model_training_lab.tracking import MLflowTracker

    tracker = MLflowTracker(no_mlflow=no_mlflow)
    run_id = tracker.log_eval_run(result)
    if run_id:
        console.print(f"  [green]MLflow run: {run_id}[/green] ({result.variant})")


def _log_all_results(
    all_results: dict[str, list[Any]],
    comparison_rows: list[Any],
    *,
    no_mlflow: bool,
) -> None:
    """Log all track results and comparison table."""
    from rhoai_model_training_lab.tracking import MLflowTracker

    tracker = MLflowTracker(no_mlflow=no_mlflow)
    run_ids: list[str] = []

    for track_name, results in all_results.items():
        for result in results:
            run_id = tracker.log_eval_run(result)
            if run_id:
                run_ids.append(run_id)
                console.print(f"  [green]MLflow run: {run_id}[/green] ({track_name}/{result.variant})")

    if comparison_rows:
        cmp_id = tracker.log_comparison(comparison_rows, parent_run_ids=run_ids)
        if cmp_id:
            console.print(f"  [green]Comparison table: {cmp_id}[/green]")


def _print_result(result: Any) -> None:
    """Print a summary of a single EvalResult."""
    console.print(f"\n[bold]{result.track} — {result.variant}[/bold]")
    console.print(f"  Status: {result.execution_status}")
    if result.metrics:
        m = result.metrics
        if m.task_success_rate is not None:
            console.print(f"  Task success rate: {m.task_success_rate:.4f}")
        if m.pass_k is not None:
            console.print(f"  pass^{m.pass_k_k}: {m.pass_k:.4f}")
        if m.answer_accuracy is not None:
            console.print(f"  Answer accuracy: {m.answer_accuracy:.4f}")
        console.print(f"  Attempted/Succeeded/Failed: {m.attempted}/{m.succeeded}/{m.failed}")
    for rr in result.retention_results:
        console.print(f"  Retention ({rr.variant}): {rr.accuracy:.4f} (Δ={rr.retention_delta_pp:+.2f} pp)")


def _print_summary(all_results: dict[str, list[Any]], comparison_rows: list[Any]) -> None:
    """Print the full evaluation summary."""
    console.rule("[bold blue]Results Summary[/bold blue]")
    for track_name, results in all_results.items():
        for result in results:
            _print_result(result)

    if comparison_rows:
        console.print("\n[bold]Comparison Table:[/bold]")
        try:
            import pandas as pd

            df = pd.DataFrame([r.model_dump() for r in comparison_rows])
            console.print(df.to_string(index=False))
        except ImportError:
            for row in comparison_rows:
                console.print(f"  {row.variant}: success={row.task_success_rate} pass^k={row.pass_k}")


# ---------------------------------------------------------------------------
# log-results
# ---------------------------------------------------------------------------

@main.command("log-results")
@click.option(
    "--category",
    type=click.Choice(["data", "training", "evaluation", "comparison", "all"]),
    default="all",
    help="Category of results to re-upload.",
)
@click.pass_context
def log_results(ctx: click.Context, category: str) -> None:
    """Re-upload locally saved results to MLflow.

    Use this after offline runs (--no-mlflow) or when MLflow logging failed
    during evaluation.
    """
    console.rule("[bold blue]Re-upload Results to MLflow[/bold blue]")

    from rhoai_model_training_lab.tracking import MLflowTracker

    tracker = MLflowTracker()

    categories = [category] if category != "all" else ["data", "training", "evaluation", "comparison"]

    total_uploaded = 0
    for cat in categories:
        console.print(f"\n[bold]Processing category: {cat}[/bold]")
        try:
            run_ids = tracker.reupload_pending(cat)
            if run_ids:
                console.print(f"  ✅ Uploaded {len(run_ids)} records")
                for rid in run_ids:
                    console.print(f"    → {rid}")
                total_uploaded += len(run_ids)
            else:
                console.print("  No pending records")
        except Exception as exc:
            console.print(f"  [red]Failed: {exc}[/red]")

    console.print(f"\n[bold]Total re-uploaded: {total_uploaded}[/bold]")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@main.command()
@click.option("--host", default="127.0.0.1", help="Server host.")
@click.option("--port", default=8000, type=int, help="Server port.")
@click.option("--reload", "auto_reload", is_flag=True, help="Enable auto-reload for development.")
@click.option("--workers", default=1, type=int, help="Number of workers.")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, auto_reload: bool, workers: int) -> None:
    """Start the RAG harness backend server."""
    console.rule("[bold blue]Starting Backend Server[/bold blue]")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  Workers: {workers}")
    console.print(f"  Auto-reload: {auto_reload}")

    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed. Install with: pip install -e '.[backend]'[/red]")
        sys.exit(1)

    try:
        from rhoai_model_training_lab.api import app  # noqa: F401

        console.print("\n[green]Starting server...[/green]")
        uvicorn.run(
            "rhoai_model_training_lab.api:app",
            host=host,
            port=port,
            reload=auto_reload,
            workers=workers,
            log_level="info",
        )
    except ImportError as exc:
        console.print(f"[red]Backend module not available: {exc}[/red]")
        console.print("Install backend dependencies: pip install -e '.[backend]'")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Server startup failed")
        console.print(f"[red]Server failed: {exc}[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
