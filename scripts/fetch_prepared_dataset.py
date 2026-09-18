#!/usr/bin/env python3
"""Fetch a prepared dataset bundle from local, S3, or PVC sources.

Usage:
    python scripts/fetch_prepared_dataset.py --release configs/data-release.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksums(checksum_file: Path) -> dict[str, str]:
    """Parse a checksums.sha256 file into {relative_path: hash}."""
    checksums: dict[str, str] = {}
    if not checksum_file.exists():
        return checksums
    for line in checksum_file.read_text().strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            checksums[parts[1].strip()] = parts[0].strip()
    return checksums


def verify_checksums(bundle_path: Path) -> tuple[bool, list[str], list[str]]:
    """Verify checksums against checksums.sha256 in the bundle.

    Returns (all_ok, mismatches, missing_files).
    """
    checksum_file = bundle_path / "checksums.sha256"
    if not checksum_file.exists():
        return False, [], ["checksums.sha256 not found"]

    expected = load_checksums(checksum_file)
    mismatches: list[str] = []
    missing: list[str] = []

    for rel_path, expected_hash in expected.items():
        file_path = bundle_path / rel_path
        if not file_path.exists():
            missing.append(rel_path)
            continue
        actual_hash = compute_sha256(file_path)
        if actual_hash != expected_hash:
            mismatches.append(f"{rel_path}: expected {expected_hash[:16]}... got {actual_hash[:16]}...")

    all_ok = len(mismatches) == 0 and len(missing) == 0
    return all_ok, mismatches, missing


def try_local(source_cfg: dict, dest: Path) -> bool:
    """Attempt to fetch from a local path."""
    src = Path(os.path.expandvars(source_cfg.get("path", "")))
    if not src.exists():
        console.print(f"  [dim]Local source not found: {src}[/dim]")
        return False
    if src.resolve() == dest.resolve():
        console.print(f"  [green]Bundle already at local path: {src}[/green]")
        return True

    console.print(f"  Copying from local: {src} → {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return True


def try_s3(source_cfg: dict, dest: Path) -> bool:
    """Attempt to fetch from S3-compatible storage."""
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError:
        console.print("  [dim]boto3 not installed — skipping S3 source[/dim]")
        return False

    bucket = os.path.expandvars(source_cfg.get("bucket", ""))
    prefix = os.path.expandvars(source_cfg.get("prefix", "")).rstrip("/")
    endpoint_url = os.environ.get("S3_ENDPOINT", "") or None
    region = os.environ.get("S3_REGION", "us-east-1")

    if not bucket:
        console.print("  [dim]S3 bucket not configured — skipping[/dim]")
        return False

    try:
        session = boto3.session.Session(
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
            region_name=region,
        )
        s3 = session.client(
            "s3",
            endpoint_url=endpoint_url,
            config=BotoConfig(signature_version="s3v4"),
        )

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix + "/")

        file_count = 0
        dest.mkdir(parents=True, exist_ok=True)
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix) :].lstrip("/")
                if not rel:
                    continue
                local_file = dest / rel
                local_file.parent.mkdir(parents=True, exist_ok=True)
                console.print(f"  [dim]Downloading s3://{bucket}/{key}[/dim]")
                s3.download_file(bucket, key, str(local_file))
                file_count += 1

        if file_count > 0:
            console.print(f"  [green]Downloaded {file_count} files from S3[/green]")
            return True
        else:
            console.print(f"  [dim]No files found at s3://{bucket}/{prefix}/[/dim]")
            return False

    except Exception as exc:
        console.print(f"  [dim]S3 fetch failed: {exc}[/dim]")
        return False


FETCH_HANDLERS = {
    "local": try_local,
    "s3": try_s3,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a prepared dataset bundle for training.",
    )
    parser.add_argument(
        "--release",
        type=str,
        default="configs/data-release.yaml",
        help="Path to data-release.yaml config (default: configs/data-release.yaml)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Override destination path (default: from config bundle.base_path)",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip checksum verification after download",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        from rhoai_model_training_lab.config import load_env, load_yaml_config

        load_env()
        config = load_yaml_config(args.release)
    except Exception as exc:
        console.print(f"[red]Failed to load config: {exc}[/red]")
        sys.exit(1)

    bundle_cfg = config.get("bundle", {})
    fetch_cfg = config.get("fetch", {})
    sources = fetch_cfg.get("sources", [])
    never_trigger_sdg = fetch_cfg.get("never_trigger_sdg", True)

    dest = Path(args.dest) if args.dest else Path(bundle_cfg.get("base_path", "data/prepared/tau-knowledge-v1"))

    console.print(f"[bold]Fetching bundle:[/bold] {bundle_cfg.get('name', 'unknown')} {bundle_cfg.get('version', '')}")
    console.print(f"[bold]Destination:[/bold] {dest}")
    console.print()

    # Try sources in order
    fetched = False
    for source in sources:
        source_type = source.get("type", "")
        handler = FETCH_HANDLERS.get(source_type)
        if handler is None:
            console.print(f"[yellow]Unknown source type: {source_type} — skipping[/yellow]")
            continue

        console.print(f"[bold]Trying source:[/bold] {source_type}")
        if handler(source, dest):
            fetched = True
            break
        console.print()

    if not fetched:
        console.print()
        console.print("[red bold]ERROR: Could not fetch the prepared dataset bundle.[/red bold]")
        console.print("[red]All configured sources were unavailable or empty.[/red]")
        if never_trigger_sdg:
            console.print(
                "[yellow]NOTE: This script will NOT trigger synthetic data generation.[/yellow]"
            )
            console.print(
                "[yellow]To create a bundle, use the authoring path:[/yellow]"
            )
            console.print("  python scripts/prepare_tau_sources.py --config configs/data-preparation.yaml")
            console.print("  python scripts/generate_synthetic.py --config configs/sdg.yaml")
            console.print("  python scripts/build_prepared_bundle.py --config configs/data-release.yaml")
        sys.exit(1)

    # Verify manifest exists
    manifest_path = dest / "manifest.json"
    if not manifest_path.exists():
        console.print("[red]ERROR: Downloaded bundle is missing manifest.json[/red]")
        sys.exit(1)

    # Verify checksums
    if args.skip_checksum:
        console.print("[yellow]Checksum verification skipped (--skip-checksum)[/yellow]")
    else:
        console.print()
        console.print("[bold]Verifying checksums...[/bold]")
        all_ok, mismatches, missing = verify_checksums(dest)

        if missing:
            console.print("[red]Missing files:[/red]")
            for f in missing:
                console.print(f"  ❌  {f}")

        if mismatches:
            console.print("[red]Checksum mismatches:[/red]")
            for m in mismatches:
                console.print(f"  ❌  {m}")

        if all_ok:
            console.print("[green]✅  All checksums verified[/green]")
        else:
            console.print("[red bold]ERROR: Checksum verification failed.[/red bold]")
            console.print("[red]The bundle may be corrupted or incomplete.[/red]")
            sys.exit(1)

    # Check for licensing conditions
    licenses_dir = dest / "LICENSES"
    dataset_card = dest / "DATASET_CARD.md"
    if licenses_dir.exists():
        license_files = list(licenses_dir.iterdir())
        if license_files:
            console.print()
            console.print(f"[yellow]⚠️  Licensing conditions apply — see {licenses_dir}/ ({len(license_files)} file(s))[/yellow]")
    if dataset_card.exists():
        console.print(f"[dim]Dataset card available at: {dataset_card}[/dim]")

    # Check version compatibility
    try:
        import json

        manifest = json.loads(manifest_path.read_text())
        bundle_model = manifest.get("model_id", "")
        env_model = os.environ.get("BASE_MODEL_ID", "")
        if env_model and bundle_model and env_model != bundle_model:
            console.print(
                f"[yellow]⚠️  Model mismatch: bundle expects '{bundle_model}', "
                f"env BASE_MODEL_ID is '{env_model}'[/yellow]"
            )
    except Exception:
        pass

    console.print()
    console.print(f"[green bold]✅  Bundle fetched successfully: {dest}[/green bold]")
    console.print("Next step: python scripts/validate_prepared_dataset.py --bundle " + str(dest))


if __name__ == "__main__":
    main()
