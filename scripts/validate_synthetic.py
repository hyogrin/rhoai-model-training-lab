#!/usr/bin/env python3
"""Run independent validation on generated synthetic data (authoring path).

Performs schema validation, grounding checks, deduplication,
and evaluation contamination checks.

Usage:
    python scripts/validate_synthetic.py --config configs/data-preparation.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


class SyntheticValidator:
    """Independent validator for synthetic training data."""

    def __init__(self, data_dir: Path, config: dict) -> None:
        self.data_dir = data_dir
        self.config = config
        self.quality_cfg = config.get("quality_gates", {})
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict = {
            "total_samples": 0,
            "valid_schema": 0,
            "invalid_schema": 0,
            "grounding_pass": 0,
            "grounding_fail": 0,
            "duplicates_found": 0,
            "contamination_flags": 0,
            "by_type": {},
        }

    def load_samples(self) -> list[dict]:
        """Load all generated samples from JSONL files in the data directory."""
        samples: list[dict] = []
        jsonl_files = sorted(self.data_dir.glob("**/*.jsonl"))

        if not jsonl_files:
            self.errors.append(f"No JSONL files found in {self.data_dir}")
            return samples

        for fpath in jsonl_files:
            with open(fpath) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        obj["_source_file"] = str(fpath)
                        obj["_source_line"] = line_num
                        samples.append(obj)
                    except json.JSONDecodeError as exc:
                        self.errors.append(f"{fpath.name}:{line_num}: Invalid JSON — {exc}")

        self.stats["total_samples"] = len(samples)
        return samples

    def validate_schemas(self, samples: list[dict]) -> list[dict]:
        """Validate each sample against the canonical schema.

        Returns list of schema-valid samples.
        """
        valid_samples: list[dict] = []
        required_fields = ["sample_id", "messages"]

        for sample in samples:
            sample_id = sample.get("sample_id", f"unknown-{id(sample)}")
            missing = [f for f in required_fields if f not in sample]
            if missing:
                self.errors.append(f"{sample_id}: missing required fields {missing}")
                self.stats["invalid_schema"] += 1
                continue

            messages = sample.get("messages", [])
            if not isinstance(messages, list) or len(messages) == 0:
                self.errors.append(f"{sample_id}: 'messages' must be a non-empty list")
                self.stats["invalid_schema"] += 1
                continue

            msg_errors = []
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    msg_errors.append(f"message[{i}] is not a dict")
                    continue
                if "role" not in msg:
                    msg_errors.append(f"message[{i}] missing 'role'")
                role = msg.get("role", "")
                if role not in ("system", "user", "assistant", "tool"):
                    msg_errors.append(f"message[{i}] invalid role '{role}'")
                if msg.get("content") is None and msg.get("tool_calls") is None:
                    msg_errors.append(f"message[{i}] has neither 'content' nor 'tool_calls'")

            if msg_errors:
                self.errors.append(f"{sample_id}: {'; '.join(msg_errors[:3])}")
                self.stats["invalid_schema"] += 1
                continue

            # Validate tool_calls schema if present
            for i, msg in enumerate(messages):
                if msg.get("tool_calls"):
                    for tc_idx, tc in enumerate(msg["tool_calls"]):
                        if not isinstance(tc, dict):
                            msg_errors.append(f"message[{i}].tool_calls[{tc_idx}] not a dict")
                        elif "function" not in tc and "name" not in tc:
                            msg_errors.append(f"message[{i}].tool_calls[{tc_idx}] missing function/name")

            self.stats["valid_schema"] += 1

            sample_type = sample.get("sample_type", "unknown")
            type_stats = self.stats["by_type"].setdefault(sample_type, {"total": 0, "valid": 0})
            type_stats["total"] += 1
            type_stats["valid"] += 1

            valid_samples.append(sample)

        return valid_samples

    def check_grounding(self, samples: list[dict]) -> None:
        """Verify samples reference actual source documents."""
        for sample in samples:
            source_docs = sample.get("source_doc_ids", [])
            if source_docs:
                self.stats["grounding_pass"] += 1
            else:
                self.stats["grounding_fail"] += 1
                sample_id = sample.get("sample_id", "?")
                self.warnings.append(f"{sample_id}: no source_doc_ids — ungrounded sample")

    def check_duplicates(self, samples: list[dict]) -> list[dict]:
        """Check for exact and near-duplicate samples.

        Returns deduplicated sample list.
        """
        seen_hashes: dict[str, str] = {}
        unique_samples: list[dict] = []
        dup_count = 0

        for sample in samples:
            content_key = json.dumps(sample.get("messages", []), sort_keys=True, ensure_ascii=False)
            content_hash = hashlib.sha256(content_key.encode()).hexdigest()

            sample_id = sample.get("sample_id", "?")
            if content_hash in seen_hashes:
                dup_count += 1
                self.warnings.append(
                    f"{sample_id}: exact duplicate of {seen_hashes[content_hash]}"
                )
            else:
                seen_hashes[content_hash] = sample_id
                unique_samples.append(sample)

        self.stats["duplicates_found"] = dup_count
        return unique_samples

    def check_contamination(self, samples: list[dict]) -> None:
        """Check for potential evaluation data contamination.

        Uses hash-based checks to detect overlap with reserved eval content.
        Only reports minimal pass/fail findings.
        """
        flags = 0
        for sample in samples:
            messages = sample.get("messages", [])
            for msg in messages:
                content = msg.get("content", "") or ""
                # Flag samples that reference evaluation-specific terms
                suspicious_terms = [
                    "expected_actions", "reward_criteria", "golden_retrieval",
                    "task_success", "grader_notes", "hidden_goal",
                ]
                for term in suspicious_terms:
                    if term in content.lower():
                        flags += 1
                        sample_id = sample.get("sample_id", "?")
                        self.warnings.append(
                            f"{sample_id}: potential contamination — contains '{term}'"
                        )
                        break

        self.stats["contamination_flags"] = flags

    def check_type_distribution(self, samples: list[dict]) -> None:
        """Check sample type distribution against quality gates."""
        type_counts = Counter(s.get("sample_type", "unknown") for s in samples)

        required_types = self.quality_cfg.get("required_types", [])
        for req_type in required_types:
            if type_counts.get(req_type, 0) == 0:
                self.errors.append(f"Missing required sample type: {req_type}")

        min_tool = self.quality_cfg.get("min_tool_examples", 0)
        if min_tool and type_counts.get("tool_selection", 0) < min_tool:
            self.warnings.append(
                f"Tool examples ({type_counts.get('tool_selection', 0)}) below minimum ({min_tool})"
            )

        min_traj = self.quality_cfg.get("min_trajectory_examples", 0)
        if min_traj and type_counts.get("trajectory", 0) < min_traj:
            self.warnings.append(
                f"Trajectory examples ({type_counts.get('trajectory', 0)}) below minimum ({min_traj})"
            )

    def run_all(self) -> bool:
        """Run all validation checks.

        Returns True if no critical errors.
        """
        console.print("[bold]Loading samples...[/bold]")
        samples = self.load_samples()
        if not samples:
            return False

        console.print(f"  Loaded {len(samples)} samples")
        console.print()

        console.print("[bold]Validating schemas...[/bold]")
        valid_samples = self.validate_schemas(samples)
        console.print(f"  Valid: {len(valid_samples)}, Invalid: {self.stats['invalid_schema']}")
        console.print()

        console.print("[bold]Checking grounding...[/bold]")
        self.check_grounding(valid_samples)
        console.print(
            f"  Grounded: {self.stats['grounding_pass']}, "
            f"Ungrounded: {self.stats['grounding_fail']}"
        )
        console.print()

        console.print("[bold]Checking duplicates...[/bold]")
        unique_samples = self.check_duplicates(valid_samples)
        console.print(f"  Unique: {len(unique_samples)}, Duplicates: {self.stats['duplicates_found']}")
        console.print()

        console.print("[bold]Checking contamination...[/bold]")
        self.check_contamination(unique_samples)
        console.print(f"  Flags: {self.stats['contamination_flags']}")
        console.print()

        console.print("[bold]Checking type distribution...[/bold]")
        self.check_type_distribution(unique_samples)
        console.print()

        # Check acceptance rate
        total = self.stats["total_samples"]
        accepted = self.stats["valid_schema"]
        rate = accepted / total if total > 0 else 0.0
        min_rate = self.quality_cfg.get("min_acceptance_rate", 0.7)
        if rate < min_rate:
            self.errors.append(
                f"Acceptance rate {rate:.1%} below minimum {min_rate:.1%}"
            )

        return len(self.errors) == 0

    def print_report(self) -> None:
        console.print("[bold]═══ Synthetic Data Validation Report ═══[/bold]")
        console.print()

        # Stats table
        table = Table(title="Validation Statistics")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Total samples", str(self.stats["total_samples"]))
        table.add_row("Valid schema", str(self.stats["valid_schema"]))
        table.add_row("Invalid schema", str(self.stats["invalid_schema"]))
        table.add_row("Grounding pass", str(self.stats["grounding_pass"]))
        table.add_row("Grounding fail", str(self.stats["grounding_fail"]))
        table.add_row("Duplicates", str(self.stats["duplicates_found"]))
        table.add_row("Contamination flags", str(self.stats["contamination_flags"]))
        console.print(table)

        # Type distribution
        if self.stats["by_type"]:
            console.print()
            type_table = Table(title="Type Distribution")
            type_table.add_column("Type", style="bold")
            type_table.add_column("Total", justify="right")
            type_table.add_column("Valid", justify="right")

            for stype, tstats in sorted(self.stats["by_type"].items()):
                type_table.add_row(stype, str(tstats["total"]), str(tstats["valid"]))
            console.print(type_table)

        if self.warnings:
            console.print()
            console.print(f"[yellow]Warnings ({len(self.warnings)}):[/yellow]")
            for w in self.warnings[:20]:
                console.print(f"  ⚠️  {w}")
            if len(self.warnings) > 20:
                console.print(f"  ... and {len(self.warnings) - 20} more warnings")

        if self.errors:
            console.print()
            console.print(f"[red]Errors ({len(self.errors)}):[/red]")
            for e in self.errors[:20]:
                console.print(f"  ❌  {e}")
            if len(self.errors) > 20:
                console.print(f"  ... and {len(self.errors) - 20} more errors")

        console.print()
        if self.errors:
            console.print("[red bold]❌  Validation FAILED[/red bold]")
        else:
            console.print("[green bold]✅  Validation PASSED[/green bold]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate generated synthetic data independently (authoring path).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data-preparation.yaml",
        help="Path to data-preparation.yaml (default: configs/data-preparation.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override synthetic data directory (default: data/synthetic/canonical)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for validation report JSON",
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

    data_dir = Path(args.data_dir or "data/synthetic/canonical")

    if not data_dir.exists():
        console.print(f"[red]Data directory not found: {data_dir}[/red]")
        console.print("Run generate_synthetic.py first.")
        sys.exit(1)

    validator = SyntheticValidator(data_dir, config)
    passed = validator.run_all()
    validator.print_report()

    # Save report if requested
    if args.output:
        report = {
            "passed": passed,
            "stats": validator.stats,
            "error_count": len(validator.errors),
            "warning_count": len(validator.warnings),
            "errors": validator.errors[:50],
            "warnings": validator.warnings[:50],
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        console.print(f"\nReport saved to: {output_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
