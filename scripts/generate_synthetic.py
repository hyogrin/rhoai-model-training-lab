#!/usr/bin/env python3
"""Run SDG pipeline to generate synthetic training data (authoring path).

Uses sdg_hub to generate policy QA, policy application, tool selection,
trajectory, and clarification examples from τ-Knowledge banking sources.

Usage:
    python scripts/generate_synthetic.py --config configs/sdg.yaml [--profile smoke|lab]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()


class BudgetTracker:
    """Track API usage budget."""

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self.total_tokens = 0
        self.total_calls = 0
        self.estimated_cost_usd = 0.0
        self.start_time = time.time()

    def record_call(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.total_calls += 1
        self.total_tokens += prompt_tokens + completion_tokens
        # Conservative cost estimate ($0.01 / 1K tokens)
        self.estimated_cost_usd = self.total_tokens * 0.00001

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.limit_usd - self.estimated_cost_usd)

    @property
    def budget_exceeded(self) -> bool:
        return self.estimated_cost_usd >= self.limit_usd

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "budget_limit_usd": self.limit_usd,
            "budget_remaining_usd": round(self.budget_remaining, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


class CheckpointManager:
    """Manage generation checkpoints for resume."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.state_file = checkpoint_path / "state.json"

    def load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"completed_types": {}, "generated_ids": []}

    def save_state(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state, indent=2))

    def save_samples(self, sample_type: str, samples: list[dict]) -> None:
        outfile = self.checkpoint_path / f"{sample_type}.jsonl"
        with open(outfile, "a") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def run_sdg_pipeline(
    config: dict,
    profile: str,
    budget: BudgetTracker,
    checkpoint: CheckpointManager,
) -> dict:
    """Run the SDG pipeline using sdg_hub.

    Returns summary statistics.
    """
    teacher_cfg = config.get("teacher", {})
    gen_cfg = config.get("generation", {})
    val_cfg = config.get("validation", {})

    target_counts = gen_cfg.get("target_counts", {}).get(profile, {})
    total_target = target_counts.get("total", 64)
    schemas = gen_cfg.get("schemas", {})
    seed = gen_cfg.get("seed", 42)

    # Load checkpoint state for resume
    state = checkpoint.load_state()
    completed = state.get("completed_types", {})

    stats = {
        "profile": profile,
        "target_total": total_target,
        "generated": 0,
        "accepted": 0,
        "rejected": 0,
        "by_type": {},
        "budget": {},
        "resumed": bool(completed),
    }

    # Try importing sdg_hub
    try:
        import sdg_hub
        console.print(f"[green]sdg_hub version: {getattr(sdg_hub, '__version__', 'unknown')}[/green]")
    except ImportError:
        console.print("[red]sdg_hub not installed. Install with: pip install -e '.[sdg]'[/red]")
        sys.exit(1)

    endpoint = os.environ.get("SDG_TEACHER_ENDPOINT") or teacher_cfg.get("endpoint", "")
    api_key = os.environ.get("SDG_TEACHER_API_KEY") or teacher_cfg.get("api_key", "")
    model = os.environ.get("SDG_TEACHER_MODEL") or teacher_cfg.get("model", "")

    if not endpoint:
        console.print("[red]SDG_TEACHER_ENDPOINT not configured.[/red]")
        console.print("[yellow]Set SDG_TEACHER_ENDPOINT in .env or the config file.[/yellow]")
        sys.exit(1)

    sample_types = ["policy_qa", "policy_application", "tool_selection", "trajectory", "clarification"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("Overall", total=total_target)

        for sample_type in sample_types:
            type_target = target_counts.get(sample_type, 0)
            if type_target == 0:
                continue

            already_done = completed.get(sample_type, 0)
            remaining = max(0, type_target - already_done)
            if remaining == 0:
                stats["by_type"][sample_type] = {"target": type_target, "generated": already_done, "accepted": already_done}
                stats["generated"] += already_done
                stats["accepted"] += already_done
                progress.advance(overall_task, already_done)
                continue

            schema_cfg = schemas.get(sample_type, {})
            flow_path = schema_cfg.get("flow", "")
            max_tokens = schema_cfg.get("max_tokens", 1024)
            temperature = schema_cfg.get("temperature", 0.7)

            type_task = progress.add_task(f"  {sample_type}", total=remaining)

            type_generated = already_done
            type_accepted = already_done
            type_rejected = 0

            for i in range(remaining):
                if budget.budget_exceeded:
                    console.print(f"[yellow]Budget limit reached (${budget.limit_usd})[/yellow]")
                    break

                try:
                    from sdg_hub import generate  # type: ignore[attr-defined]

                    result = generate(
                        flow=flow_path,
                        endpoint=endpoint,
                        api_key=api_key,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        seed=seed + i + already_done,
                        sample_type=sample_type,
                    )

                    budget.record_call(
                        prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                        completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                    )

                    if result.get("status") == "accepted":
                        type_accepted += 1
                        samples = result.get("samples", [result])
                        checkpoint.save_samples(sample_type, samples)
                    else:
                        type_rejected += 1

                    type_generated += 1

                except Exception as exc:
                    console.print(f"[yellow]Generation error ({sample_type}): {exc}[/yellow]")
                    type_rejected += 1
                    type_generated += 1

                progress.advance(type_task, 1)
                progress.advance(overall_task, 1)

            stats["by_type"][sample_type] = {
                "target": type_target,
                "generated": type_generated,
                "accepted": type_accepted,
                "rejected": type_rejected,
            }
            stats["generated"] += type_generated
            stats["accepted"] += type_accepted
            stats["rejected"] += type_rejected

            # Update checkpoint
            completed[sample_type] = type_accepted
            state["completed_types"] = completed
            checkpoint.save_state(state)

    stats["budget"] = budget.summary()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data using sdg_hub (authoring path).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sdg.yaml",
        help="Path to SDG configuration (default: configs/sdg.yaml)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["smoke", "lab"],
        default="smoke",
        help="Generation profile: smoke (64 samples) or lab (3000 samples)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        from rhoai_model_training_lab.config import load_env, load_yaml_config

        load_env()
        config = load_yaml_config(args.config)
    except Exception as exc:
        console.print(f"[red]Failed to load config: {exc}[/red]")
        sys.exit(1)

    pipeline_cfg = config.get("pipeline", {})
    teacher_cfg = config.get("teacher", {})
    output_cfg = config.get("output", {})

    canonical_path = Path(args.output_dir or output_cfg.get("canonical_path", "data/synthetic/canonical"))
    rejected_path = Path(output_cfg.get("rejected_path", "data/synthetic/rejected"))
    logs_path = Path(output_cfg.get("logs_path", "data/synthetic/logs"))
    checkpoint_path = Path(pipeline_cfg.get("checkpoint_path", "data/checkpoints/sdg"))

    for d in (canonical_path, rejected_path, logs_path, checkpoint_path):
        d.mkdir(parents=True, exist_ok=True)

    console.print("[bold]═══ Synthetic Data Generation ═══[/bold]")
    console.print(f"Pipeline: {pipeline_cfg.get('name', 'unknown')} v{pipeline_cfg.get('version', '?')}")
    console.print(f"Profile: {args.profile}")
    console.print(f"Resume: {args.resume}")
    console.print()

    budget_limit = teacher_cfg.get("budget_limit_usd", 50.0)
    budget = BudgetTracker(limit_usd=budget_limit)
    checkpoint = CheckpointManager(checkpoint_path)

    if not args.resume:
        # Clear previous checkpoint
        state_file = checkpoint_path / "state.json"
        if state_file.exists():
            state_file.unlink()

    # Run pipeline
    stats = run_sdg_pipeline(config, args.profile, budget, checkpoint)

    # Save usage accounting
    usage_path = Path(output_cfg.get("usage_accounting_path", "data/synthetic/usage.json"))
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    usage_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "profile": args.profile,
        "budget": stats["budget"],
        "by_type": stats["by_type"],
    }
    usage_path.write_text(json.dumps(usage_data, indent=2))

    # Save generation log
    log_file = logs_path / f"generation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    log_file.write_text(json.dumps(stats, indent=2))

    # Print summary
    console.print()
    console.print("[bold]═══ Generation Summary ═══[/bold]")
    table = Table()
    table.add_column("Type", style="bold")
    table.add_column("Target", justify="right")
    table.add_column("Generated", justify="right")
    table.add_column("Accepted", justify="right")
    table.add_column("Rejected", justify="right")
    table.add_column("Rate", justify="right")

    for sample_type, type_stats in stats.get("by_type", {}).items():
        gen = type_stats.get("generated", 0)
        acc = type_stats.get("accepted", 0)
        rate = f"{acc / gen * 100:.0f}%" if gen > 0 else "N/A"
        table.add_row(
            sample_type,
            str(type_stats.get("target", 0)),
            str(gen),
            str(acc),
            str(type_stats.get("rejected", 0)),
            rate,
        )

    table.add_row(
        "[bold]Total[/bold]",
        str(stats.get("target_total", 0)),
        str(stats.get("generated", 0)),
        str(stats.get("accepted", 0)),
        str(stats.get("rejected", 0)),
        f"{stats['accepted'] / stats['generated'] * 100:.0f}%" if stats.get("generated", 0) > 0 else "N/A",
    )
    console.print(table)

    budget_info = stats.get("budget", {})
    console.print(f"\nBudget: ${budget_info.get('estimated_cost_usd', 0):.4f} / ${budget_info.get('budget_limit_usd', 0):.2f}")
    console.print(f"API calls: {budget_info.get('total_calls', 0)}, tokens: {budget_info.get('total_tokens', 0)}")
    console.print(f"Elapsed: {budget_info.get('elapsed_seconds', 0):.1f}s")

    console.print()
    console.print("[green]✅  Synthetic data generation complete.[/green]")
    console.print(f"Output: {canonical_path}")
    console.print("Next step: python scripts/validate_synthetic.py --config configs/data-preparation.yaml")


if __name__ == "__main__":
    main()
