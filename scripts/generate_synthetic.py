#!/usr/bin/env python3
"""Run SDG pipeline to generate synthetic training data (authoring path).

Uses sdg_hub Knowledge Tuning flows to generate QA pairs from
τ-Knowledge banking_knowledge KB documents.

Usage:
    python scripts/generate_synthetic.py --config configs/sdg.yaml [--profile smoke|lab]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# sdg_hub flow IDs for Knowledge Tuning (English)
# key_facts: atomic facts → QA pairs — most reliable with diverse teacher models
KNOWLEDGE_FLOWS = {
    "key_facts": "heavy-heart-77",
}


def load_kb_documents(sources_path: Path) -> pd.DataFrame:
    """Load KB documents from JSONL into a DataFrame for sdg_hub."""
    kb_path = sources_path / "kb" / "documents.jsonl"
    if not kb_path.exists():
        console.print(f"[red]KB documents not found: {kb_path}[/red]")
        console.print("Run notebook 01_prepare_tau_sources first.")
        sys.exit(1)

    docs = []
    with open(kb_path) as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    if not docs:
        console.print("[red]No documents in KB snapshot.[/red]")
        sys.exit(1)

    df = pd.DataFrame(docs)

    # Map to sdg_hub expected columns
    if "content" in df.columns and "document" not in df.columns:
        df["document"] = df["content"]
    if "title" in df.columns and "document_outline" not in df.columns:
        df["document_outline"] = df["title"]

    # Add domain column
    df["domain"] = "banking_knowledge"

    console.print(f"[green]Loaded {len(df)} KB documents[/green]")
    return df


def prepare_icl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add in-context learning columns required by some flows."""
    if len(df) < 2:
        return df

    sample_doc = df["document"].iloc[0] if "document" in df.columns else ""
    df["icl_document"] = sample_doc
    df["icl_query_1"] = "What are the main policies described in this document?"
    df["icl_query_2"] = "What conditions or exceptions apply?"
    df["icl_query_3"] = "How does this policy affect customer accounts?"
    return df


def run_flow(
    flow_name: str,
    flow_id: str,
    df: pd.DataFrame,
    model: str,
    api_base: str,
    api_key: str,
    max_samples: int,
    checkpoint_dir: str | None = None,
    max_concurrency: int = 4,
) -> pd.DataFrame | None:
    """Run a single sdg_hub flow and return results."""
    from sdg_hub import Flow, FlowRegistry

    fr = FlowRegistry()
    flow_path = fr.get_flow_path(flow_id)
    if not flow_path:
        console.print(f"[yellow]Flow {flow_id} not found in registry[/yellow]")
        return None

    flow = Flow.from_yaml(flow_path)

    # Limit input samples
    input_df = df.head(max_samples).copy()

    # Check required columns and add ICL if needed
    reqs = flow.get_dataset_requirements()
    for col in reqs.required_columns:
        if col not in input_df.columns:
            if col.startswith("icl_"):
                input_df = prepare_icl_columns(input_df)
                break

    # Missing columns check
    missing = [c for c in reqs.required_columns if c not in input_df.columns]
    if missing:
        console.print(f"[yellow]Missing columns for {flow_name}: {missing} — skipping[/yellow]")
        return None

    # Configure model — litellm requires provider prefix (e.g. openai/model-name)
    litellm_model = model if "/" in model else f"openai/{model}"
    flow.set_model_config(
        model=litellm_model,
        api_base=api_base,
        api_key=api_key,
    )

    console.print(f"  Running {flow_name} on {len(input_df)} documents...")
    try:
        result_df = flow.generate(
            dataset=input_df,
            checkpoint_dir=checkpoint_dir,
            max_concurrency=max_concurrency,
        )
        console.print(f"  [green]✅ {flow_name}: {len(result_df)} samples generated[/green]")
        return result_df
    except Exception as exc:
        console.print(f"  [red]❌ {flow_name} failed: {exc}[/red]")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data using sdg_hub (authoring path).",
    )
    parser.add_argument(
        "--config", type=str, default="configs/sdg.yaml",
        help="Path to SDG configuration",
    )
    parser.add_argument(
        "--profile", type=str, choices=["smoke", "lab"], default="smoke",
        help="Generation profile: smoke (quick) or lab (full)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    try:
        from rhoai_model_training_lab.config import load_env, load_yaml_config, PROJECT_ROOT
        load_env()
        config = load_yaml_config(args.config)
    except Exception as exc:
        console.print(f"[red]Failed to load config: {exc}[/red]")
        sys.exit(1)

    # Endpoints
    endpoint = os.environ.get("SDG_TEACHER_ENDPOINT", "")
    api_key = os.environ.get("SDG_TEACHER_API_KEY", "")
    model = os.environ.get("SDG_TEACHER_MODEL", "")

    if not endpoint:
        console.print("[red]SDG_TEACHER_ENDPOINT not set.[/red]")
        sys.exit(1)

    # Need api_base in OpenAI-compatible format: endpoint + /v1
    api_base = endpoint.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base += "/v1"

    # Output paths
    output_cfg = config.get("output", {})
    canonical_path = PROJECT_ROOT / output_cfg.get("canonical_path", "data/synthetic/canonical")
    logs_path = PROJECT_ROOT / output_cfg.get("logs_path", "data/synthetic/logs")
    checkpoint_base = PROJECT_ROOT / config.get("pipeline", {}).get("checkpoint_path", "data/checkpoints/sdg")

    for d in (canonical_path, logs_path, checkpoint_base):
        d.mkdir(parents=True, exist_ok=True)

    # Load KB documents
    sources_path = PROJECT_ROOT / "data" / "sources"
    df = load_kb_documents(sources_path)

    # Profile determines how many docs to process per flow
    teacher_cfg = config.get("teacher", {})
    max_concurrent = teacher_cfg.get("max_concurrent", 4)

    profile_doc_limits = {
        "smoke": 5,       # ~300 QA pairs — pipeline validation
        "lab": 20,        # ~1,500 QA pairs — suitable for LoRA/OSFT lab training
    }
    docs_per_flow = min(profile_doc_limits.get(args.profile, 20), len(df))

    console.print(f"\n[bold]═══ Synthetic Data Generation ({args.profile}) ═══[/bold]")
    console.print(f"Model: {model}")
    console.print(f"Documents per flow: {docs_per_flow}")
    console.print(f"Flows: {list(KNOWLEDGE_FLOWS.keys())}")
    console.print()

    start_time = time.time()
    all_results: dict[str, pd.DataFrame] = {}

    for flow_name, flow_id in KNOWLEDGE_FLOWS.items():
        out_file = canonical_path / f"{flow_name}.jsonl"

        # Skip if output already exists (idempotent)
        if out_file.exists() and out_file.stat().st_size > 0 and not args.resume:
            existing = sum(1 for line in open(out_file) if line.strip())
            console.print(f"  [cyan]⏭  {flow_name}: {existing}개 이미 존재 — 건너뜁니다 (재생성: --resume)[/cyan]")
            all_results[flow_name] = pd.read_json(out_file, lines=True)
            continue

        # Always use checkpoint dir for intermediate result protection
        checkpoint_dir = str(checkpoint_base / flow_name)

        result = run_flow(
            flow_name=flow_name,
            flow_id=flow_id,
            df=df,
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_samples=docs_per_flow,
            checkpoint_dir=checkpoint_dir,
            max_concurrency=max_concurrent,
        )
        if result is not None and len(result) > 0:
            all_results[flow_name] = result
            result.to_json(out_file, orient="records", lines=True, force_ascii=False)

    elapsed = time.time() - start_time

    # Summary
    console.print(f"\n[bold]═══ Generation Summary ═══[/bold]")
    table = Table()
    table.add_column("Flow", style="bold")
    table.add_column("Samples", justify="right")
    table.add_column("Output")

    total_samples = 0
    for flow_name, result_df in all_results.items():
        n = len(result_df)
        total_samples += n
        table.add_row(flow_name, str(n), str(canonical_path / f"{flow_name}.jsonl"))

    table.add_row("[bold]Total[/bold]", f"[bold]{total_samples}[/bold]", "")
    console.print(table)
    console.print(f"\nElapsed: {elapsed:.1f}s")

    # Save usage log
    usage_path = PROJECT_ROOT / output_cfg.get("usage_accounting_path", "data/synthetic/usage.json")
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    usage_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "total_samples": total_samples,
        "elapsed_seconds": round(elapsed, 1),
        "flows": {k: len(v) for k, v in all_results.items()},
    }
    usage_path.write_text(json.dumps(usage_data, indent=2))

    if total_samples > 0:
        console.print(f"\n[green]✅ Generation complete: {total_samples} samples[/green]")
    else:
        console.print("\n[red]⚠️ No samples generated. Check teacher endpoint and logs.[/red]")

    console.print("Next: python scripts/validate_synthetic.py --config configs/data-preparation.yaml")


if __name__ == "__main__":
    main()
