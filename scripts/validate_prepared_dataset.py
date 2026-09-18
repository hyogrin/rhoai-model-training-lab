#!/usr/bin/env python3
"""Validate a prepared dataset bundle for completeness and integrity.

Usage:
    python scripts/validate_prepared_dataset.py --bundle data/prepared/tau-knowledge-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


REQUIRED_FILES = [
    "manifest.json",
    "checksums.sha256",
    "canonical/train.jsonl",
    "canonical/validation.jsonl",
    "training/lora/train.jsonl",
    "training/lora/validation.jsonl",
    "training/osft/train.jsonl",
    "training/osft/validation.jsonl",
    "kb/documents.jsonl",
]

OPTIONAL_FILES = [
    "examples/train_examples.jsonl",
    "metadata/provenance.jsonl",
    "metadata/splits.json",
    "reports/quality.json",
    "reports/quality.md",
    "reports/backend-validation.json",
    "DATASET_CARD.md",
    "configs/lora.yaml",
    "configs/osft.yaml",
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: Path) -> int:
    """Count valid JSON lines in a JSONL file."""
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    pass
    return count


def validate_jsonl_schema(path: Path, required_fields: list[str]) -> tuple[int, list[str]]:
    """Validate each JSON line has required fields. Returns (valid_count, errors)."""
    errors: list[str] = []
    valid = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {i}: invalid JSON — {exc}")
                continue
            missing = [fld for fld in required_fields if fld not in obj]
            if missing:
                errors.append(f"Line {i}: missing fields {missing}")
            else:
                valid += 1
    return valid, errors


class BundleValidator:
    """Validates a prepared dataset bundle."""

    def __init__(self, bundle_path: Path) -> None:
        self.bundle_path = bundle_path
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []
        self.manifest: dict = {}

    def check_exists(self) -> bool:
        if not self.bundle_path.exists():
            self.errors.append(f"Bundle path does not exist: {self.bundle_path}")
            return False
        if not self.bundle_path.is_dir():
            self.errors.append(f"Bundle path is not a directory: {self.bundle_path}")
            return False
        return True

    def check_required_files(self) -> None:
        for rel in REQUIRED_FILES:
            p = self.bundle_path / rel
            if not p.exists():
                self.errors.append(f"Missing required file: {rel}")
            elif p.stat().st_size == 0:
                self.errors.append(f"Empty required file: {rel}")
            else:
                self.info.append(f"Found: {rel}")

        for rel in OPTIONAL_FILES:
            p = self.bundle_path / rel
            if p.exists():
                self.info.append(f"Found optional: {rel}")
            else:
                self.warnings.append(f"Missing optional file: {rel}")

    def check_manifest(self) -> None:
        manifest_path = self.bundle_path / "manifest.json"
        if not manifest_path.exists():
            return

        try:
            self.manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            self.errors.append(f"Cannot parse manifest.json: {exc}")
            return

        required_keys = [
            "bundle_name", "bundle_version", "model_id",
            "canonical_train_count", "canonical_validation_count",
        ]
        for key in required_keys:
            if key not in self.manifest:
                self.errors.append(f"Manifest missing key: {key}")

        self.info.append(
            f"Manifest: {self.manifest.get('bundle_name', '?')} "
            f"v{self.manifest.get('bundle_version', '?')} "
            f"model={self.manifest.get('model_id', '?')}"
        )

    def check_checksums(self) -> None:
        checksum_file = self.bundle_path / "checksums.sha256"
        if not checksum_file.exists():
            return

        expected: dict[str, str] = {}
        for line in checksum_file.read_text().strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                expected[parts[1].strip()] = parts[0].strip()

        verified = 0
        for rel_path, expected_hash in expected.items():
            file_path = self.bundle_path / rel_path
            if not file_path.exists():
                self.errors.append(f"Checksum file missing: {rel_path}")
                continue
            actual = compute_sha256(file_path)
            if actual != expected_hash:
                self.errors.append(
                    f"Checksum mismatch: {rel_path} "
                    f"(expected {expected_hash[:16]}..., got {actual[:16]}...)"
                )
            else:
                verified += 1

        self.info.append(f"Checksums: {verified}/{len(expected)} verified")

    def check_sample_counts(self) -> None:
        """Verify sample counts match the manifest."""
        if not self.manifest:
            return

        expected_train = self.manifest.get("canonical_train_count", 0)
        expected_val = self.manifest.get("canonical_validation_count", 0)

        files_to_check = {
            "canonical/train.jsonl": expected_train,
            "canonical/validation.jsonl": expected_val,
        }

        for rel, expected in files_to_check.items():
            p = self.bundle_path / rel
            if not p.exists():
                continue
            actual = count_jsonl(p)
            if expected and actual != expected:
                self.errors.append(
                    f"Sample count mismatch in {rel}: "
                    f"manifest says {expected}, found {actual}"
                )
            else:
                self.info.append(f"{rel}: {actual} samples")

    def check_canonical_schema(self) -> None:
        """Validate canonical JSONL schema."""
        canonical_fields = ["sample_id", "messages"]

        for split in ("train", "validation"):
            p = self.bundle_path / f"canonical/{split}.jsonl"
            if not p.exists():
                continue
            valid, errors = validate_jsonl_schema(p, canonical_fields)
            if errors:
                for err in errors[:5]:
                    self.errors.append(f"canonical/{split}.jsonl: {err}")
                if len(errors) > 5:
                    self.errors.append(f"canonical/{split}.jsonl: ... and {len(errors) - 5} more errors")
            self.info.append(f"canonical/{split}.jsonl: {valid} valid samples")

    def check_backend_parity(self) -> None:
        """Check that LoRA and OSFT exports have the same sample counts."""
        counts: dict[str, dict[str, int]] = {}

        for backend in ("lora", "osft"):
            counts[backend] = {}
            for split in ("train", "validation"):
                p = self.bundle_path / f"training/{backend}/{split}.jsonl"
                if p.exists():
                    counts[backend][split] = count_jsonl(p)
                else:
                    counts[backend][split] = -1

        for split in ("train", "validation"):
            lora_count = counts.get("lora", {}).get(split, -1)
            osft_count = counts.get("osft", {}).get(split, -1)

            if lora_count == -1 or osft_count == -1:
                continue

            if lora_count != osft_count:
                self.errors.append(
                    f"Backend parity mismatch ({split}): "
                    f"LoRA has {lora_count}, OSFT has {osft_count}"
                )
            else:
                self.info.append(
                    f"Backend parity OK ({split}): {lora_count} samples each"
                )

        # Also check against canonical
        for split in ("train", "validation"):
            canon_path = self.bundle_path / f"canonical/{split}.jsonl"
            if not canon_path.exists():
                continue
            canon_count = count_jsonl(canon_path)

            for backend in ("lora", "osft"):
                backend_count = counts.get(backend, {}).get(split, -1)
                if backend_count == -1:
                    continue
                if backend_count != canon_count:
                    self.warnings.append(
                        f"Canonical vs {backend} count differs ({split}): "
                        f"canonical={canon_count}, {backend}={backend_count}"
                    )

    def check_kb_documents(self) -> None:
        """Validate KB documents JSONL."""
        p = self.bundle_path / "kb/documents.jsonl"
        if not p.exists():
            return
        valid, errors = validate_jsonl_schema(p, ["text"])
        if errors:
            for err in errors[:3]:
                self.warnings.append(f"kb/documents.jsonl: {err}")
        self.info.append(f"kb/documents.jsonl: {valid} documents")

    def run_all(self) -> bool:
        if not self.check_exists():
            return False

        self.check_required_files()
        self.check_manifest()
        self.check_checksums()
        self.check_sample_counts()
        self.check_canonical_schema()
        self.check_backend_parity()
        self.check_kb_documents()

        return len(self.errors) == 0

    def print_report(self) -> None:
        console.print()
        console.print(f"[bold]Bundle Validation Report: {self.bundle_path}[/bold]")
        console.print("=" * 60)

        if self.info:
            console.print()
            console.print("[bold cyan]Info:[/bold cyan]")
            for msg in self.info:
                console.print(f"  ℹ️  {msg}")

        if self.warnings:
            console.print()
            console.print("[bold yellow]Warnings:[/bold yellow]")
            for msg in self.warnings:
                console.print(f"  ⚠️  {msg}")

        if self.errors:
            console.print()
            console.print("[bold red]Errors:[/bold red]")
            for msg in self.errors:
                console.print(f"  ❌  {msg}")

        # Summary table
        console.print()
        table = Table(title="Validation Summary")
        table.add_column("Category", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Status")

        table.add_row("Errors", str(len(self.errors)), "❌ FAIL" if self.errors else "✅ PASS")
        table.add_row("Warnings", str(len(self.warnings)), "⚠️" if self.warnings else "✅")
        table.add_row("Info", str(len(self.info)), "ℹ️")

        console.print(table)

        if self.errors:
            console.print("[red bold]❌  Bundle validation FAILED[/red bold]")
        else:
            console.print("[green bold]✅  Bundle validation PASSED[/green bold]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a prepared dataset bundle.",
    )
    parser.add_argument(
        "--bundle",
        type=str,
        required=True,
        help="Path to the prepared dataset bundle directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    validator = BundleValidator(bundle_path)
    passed = validator.run_all()

    if args.strict and validator.warnings:
        passed = False

    validator.print_report()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
