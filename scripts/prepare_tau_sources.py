#!/usr/bin/env python3
"""Prepare τ-Knowledge sources for synthetic data generation (authoring path).

Inspects the τ-bench banking_knowledge domain, extracts KB snapshots,
policy facts, tool schemas, and reserves official eval tasks.

Usage:
    python scripts/prepare_tau_sources.py --config configs/data-preparation.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def inspect_tau_bench(tau_cfg: dict) -> dict:
    """Inspect τ-bench installation and extract domain metadata.

    Uses the τ2 domain-specific API (get_knowledge_base, get_tasks)
    rather than a generic load_domain call.

    Returns a dict with kb, tasks, tools info and raw objects.
    """
    version = tau_cfg.get("version", "")
    domain = tau_cfg.get("domain", "banking_knowledge")

    info: dict = {
        "version": version,
        "domain": domain,
        "kb_found": False,
        "policies_found": False,
        "tools_found": False,
        "tasks_found": False,
        "task_count": 0,
    }

    try:
        from tau2.domains.banking_knowledge import get_knowledge_base, get_tasks
        from tau2.domains.banking_knowledge.tools import KnowledgeTools

        kb = get_knowledge_base()
        if kb is not None:
            docs = kb.get_all_documents() if hasattr(kb, "get_all_documents") else kb.documents
            info["kb_found"] = True
            info["kb"] = kb
            info["kb_documents"] = docs
            console.print(f"[green]KB loaded: {len(docs)} documents[/green]")

        tasks = get_tasks()
        if tasks:
            info["tasks_found"] = True
            info["task_count"] = len(tasks)
            info["tasks"] = tasks
            console.print(f"[green]Tasks loaded: {len(tasks)} tasks[/green]")

        tool_names = [
            a for a in dir(KnowledgeTools)
            if not a.startswith("_") and callable(getattr(KnowledgeTools, a, None))
        ]
        if tool_names:
            info["tools_found"] = True
            info["tool_names"] = tool_names
            console.print(f"[green]Tools found: {len(tool_names)} tools[/green]")

    except ImportError:
        console.print("[yellow]Could not import τ2 banking_knowledge domain[/yellow]")
    except Exception as exc:
        console.print(f"[yellow]τ-bench domain inspection failed: {exc}[/yellow]")

    return info


def extract_kb_snapshot(domain_info: dict, output_path: Path) -> int:
    """Extract and save KB documents to JSONL.

    Returns the number of documents extracted.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kb_docs = domain_info.get("kb_documents")
    if not kb_docs:
        console.print("[yellow]No KB documents available — writing empty snapshot[/yellow]")
        output_path.write_text("")
        return 0

    documents: list[dict] = []
    for doc in kb_docs:
        if isinstance(doc, dict):
            documents.append(doc)
        elif hasattr(doc, "model_dump"):
            documents.append(doc.model_dump())
        elif hasattr(doc, "__dict__"):
            documents.append(doc.__dict__)
        else:
            documents.append({"text": str(doc)})

    with open(output_path, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return len(documents)


def extract_policy_facts(domain_info: dict, output_path: Path) -> int:
    """Extract policy facts from KB documents.

    In banking_knowledge, policies are embedded within KB documents.
    We extract documents whose content contains policy-like keywords.

    Returns the number of policy facts extracted.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kb_docs = domain_info.get("kb_documents")
    if not kb_docs:
        output_path.write_text("")
        return 0

    policy_keywords = ["policy", "rule", "requirement", "must", "shall", "prohibited", "allowed"]

    facts: list[dict] = []
    for doc in kb_docs:
        if hasattr(doc, "model_dump"):
            d = doc.model_dump()
        elif hasattr(doc, "__dict__"):
            d = doc.__dict__
        elif isinstance(doc, dict):
            d = doc
        else:
            continue

        content = d.get("content", d.get("text", "")).lower()
        if any(kw in content for kw in policy_keywords):
            d.setdefault("policy_id", d.get("id", f"policy-{len(facts):04d}"))
            facts.append(d)

    with open(output_path, "w") as f:
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    return len(facts)


def extract_tool_schemas(domain_info: dict, output_path: Path) -> int:
    """Extract tool schemas from the domain.

    Returns the number of tool schemas extracted.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tool_names = domain_info.get("tool_names")
    if not tool_names:
        output_path.write_text("")
        return 0

    schemas: list[dict] = []
    for name in tool_names:
        schemas.append({"name": name})

    with open(output_path, "w") as f:
        for schema in schemas:
            f.write(json.dumps(schema, ensure_ascii=False) + "\n")

    return len(schemas)


def reserve_eval_tasks(domain_info: dict, splits_cfg: dict, output_path: Path) -> dict:
    """Reserve official eval tasks and prepare training scenario boundaries.

    Returns split statistics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = domain_info.get("tasks", [])

    result = {
        "total_tasks": len(tasks),
        "eval_reserved": len(tasks),
        "train_eligible": 0,
        "method": splits_cfg.get("method", "scenario_family"),
    }

    if not tasks:
        console.print("[yellow]No tasks found — all training data will come from SDG[/yellow]")
    else:
        console.print(
            f"[bold]Task reservation:[/bold] {result['total_tasks']} total, "
            f"{result['eval_reserved']} reserved for eval, "
            f"{result['train_eligible']} available for training seeds"
        )
        console.print("[dim]All official tasks reserved for evaluation. Training uses SDG from KB + independent scenarios.[/dim]")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare τ-Knowledge sources for SDG (authoring path).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data-preparation.yaml",
        help="Path to data-preparation.yaml (default: configs/data-preparation.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: from config output.base_path)",
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

    tau_cfg = config.get("tau_bench", {})
    sources_cfg = config.get("sources", {})
    splits_cfg = config.get("splits", {})
    output_cfg = config.get("output", {})
    base_path = Path(args.output_dir or output_cfg.get("base_path", "data/sources"))

    console.print("[bold]═══ τ-Knowledge Source Preparation ═══[/bold]")
    console.print(f"Domain: {tau_cfg.get('domain', 'banking_knowledge')}")
    console.print(f"Version: {tau_cfg.get('version', 'N/A')}")
    console.print()

    # Step 1: Inspect τ-bench
    console.print("[bold]Step 1: Inspecting τ-bench...[/bold]")
    domain_info = inspect_tau_bench(tau_cfg)

    table = Table(title="τ-bench Inspection")
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_row("KB", "✅ Found" if domain_info["kb_found"] else "❌ Not found")
    table.add_row("Policies", "✅ Found" if domain_info["policies_found"] else "❌ Not found")
    table.add_row("Tools", "✅ Found" if domain_info["tools_found"] else "❌ Not found")
    table.add_row("Tasks", f"✅ {domain_info['task_count']} tasks" if domain_info["tasks_found"] else "❌ Not found")
    console.print(table)
    console.print()

    # Step 2: Extract KB snapshot
    console.print("[bold]Step 2: Extracting KB snapshot...[/bold]")
    kb_output = Path(sources_cfg.get("kb_snapshot", {}).get("output_path", base_path / "kb" / "documents.jsonl"))
    kb_count = extract_kb_snapshot(domain_info, kb_output)
    console.print(f"  Extracted {kb_count} KB documents → {kb_output}")
    console.print()

    # Step 3: Extract policy facts
    console.print("[bold]Step 3: Extracting policy facts...[/bold]")
    policy_output = Path(sources_cfg.get("policy_extraction", {}).get("output_path", base_path / "policies" / "extracted.jsonl"))
    policy_count = extract_policy_facts(domain_info, policy_output)
    console.print(f"  Extracted {policy_count} policy facts → {policy_output}")
    console.print()

    # Step 4: Extract tool schemas
    console.print("[bold]Step 4: Extracting tool schemas...[/bold]")
    tools_output = base_path / "tools" / "schemas.jsonl"
    tool_count = extract_tool_schemas(domain_info, tools_output)
    console.print(f"  Extracted {tool_count} tool schemas → {tools_output}")
    console.print()

    # Step 5: Reserve eval tasks and prepare training boundaries
    console.print("[bold]Step 5: Reserving eval tasks...[/bold]")
    splits_output = base_path / "splits" / "task_reservation.json"
    split_stats = reserve_eval_tasks(domain_info, splits_cfg, splits_output)
    console.print()

    # Summary
    console.print("[bold]═══ Source Preparation Summary ═══[/bold]")
    summary = Table()
    summary.add_column("Output", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_column("Path")
    summary.add_row("KB documents", str(kb_count), str(kb_output))
    summary.add_row("Policy facts", str(policy_count), str(policy_output))
    summary.add_row("Tool schemas", str(tool_count), str(tools_output))
    summary.add_row("Eval tasks reserved", str(split_stats["eval_reserved"]), str(splits_output))
    console.print(summary)

    if kb_count == 0 and policy_count == 0:
        console.print()
        console.print("[red bold]⚠️  No sources extracted. Verify τ-bench installation and configuration.[/red bold]")
        sys.exit(1)

    console.print()
    console.print("[green]✅  Source preparation complete.[/green]")
    console.print("Next step: python scripts/generate_synthetic.py --config configs/sdg.yaml")


if __name__ == "__main__":
    main()
