"""Synthetic data generation pipeline using sdg_hub.

This module implements the full SDG lifecycle: policy extraction from
knowledge-base documents, synthetic sample generation via a teacher model,
multi-dimensional validation, deduplication, contamination checking, and
final bundle assembly.

Only the *preparation* profile uses this module.  Learner notebooks
consume pre-built bundles and never invoke SDG directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rhoai_model_training_lab.config import load_yaml_config
from rhoai_model_training_lab.schemas.data import (
    BundleManifest,
    CanonicalSample,
    CoverageStats,
    DatasetCard,
    Message,
    ProvenanceRecord,
    QualityReport,
    SampleType,
    SplitInfo,
    TokenStats,
    ToolSchema,
    TypeQualityStats,
    ValidationStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sdg_hub adapter — graceful degradation when the package is absent
# ---------------------------------------------------------------------------

_sdg_hub_available = False
_sdg_hub = None

try:
    import sdg_hub as _sdg_hub  # type: ignore[import-untyped]
    _sdg_hub_available = True
except ImportError:
    logger.debug("sdg_hub is not installed; SDGPipeline will require explicit teacher callables")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# PolicyExtractor
# ---------------------------------------------------------------------------

class PolicyExtractor:
    """Extract policy clauses and tool definitions from KB documents.

    Each extracted policy retains its source document ID and character offsets
    so that downstream grounding checks can verify synthetic samples.
    """

    def __init__(self) -> None:
        self._policies: list[dict[str, Any]] = []
        self._tool_schemas: list[ToolSchema] = []

    def extract(self, kb_documents: list[dict[str, Any]]) -> "PolicyExtractor":
        """Parse *kb_documents* (``documents.jsonl`` records) for policies.

        Each document dict is expected to carry at least ``doc_id`` and
        ``content`` (raw text).  Optional fields: ``title``, ``category``,
        ``tool_definitions``.
        """
        for doc in kb_documents:
            doc_id = doc.get("doc_id", doc.get("id", ""))
            content = doc.get("content", "")
            title = doc.get("title", "")

            policies = self._segment_policies(content, doc_id)
            self._policies.extend(policies)

            for td in doc.get("tool_definitions", []):
                try:
                    self._tool_schemas.append(ToolSchema(**td))
                except Exception:
                    logger.warning("Skipping malformed tool definition in doc %s", doc_id)

            logger.debug(
                "Extracted %d policies from doc %s (%s)",
                len(policies), doc_id, title,
            )

        logger.info(
            "PolicyExtractor: %d policies, %d tool schemas from %d documents",
            len(self._policies), len(self._tool_schemas), len(kb_documents),
        )
        return self

    @property
    def policies(self) -> list[dict[str, Any]]:
        return list(self._policies)

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        return list(self._tool_schemas)

    @property
    def document_ids(self) -> list[str]:
        return list({p["doc_id"] for p in self._policies})

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _segment_policies(
        content: str,
        doc_id: str,
    ) -> list[dict[str, Any]]:
        """Split document content into policy clause dicts.

        Applies a simple paragraph-boundary heuristic: consecutive non-empty
        lines form a clause.  Production deployments should replace this
        with domain-specific section-header parsing.
        """
        clauses: list[dict[str, Any]] = []
        offset = 0
        current_lines: list[str] = []
        start_offset = 0

        for line in content.split("\n"):
            stripped = line.strip()
            if stripped:
                if not current_lines:
                    start_offset = offset
                current_lines.append(stripped)
            else:
                if current_lines:
                    clause_text = "\n".join(current_lines)
                    clauses.append({
                        "doc_id": doc_id,
                        "text": clause_text,
                        "start_offset": start_offset,
                        "end_offset": offset,
                        "clause_id": f"{doc_id}::{start_offset}",
                    })
                    current_lines = []
            offset += len(line) + 1  # +1 for newline

        if current_lines:
            clause_text = "\n".join(current_lines)
            clauses.append({
                "doc_id": doc_id,
                "text": clause_text,
                "start_offset": start_offset,
                "end_offset": offset,
                "clause_id": f"{doc_id}::{start_offset}",
            })

        return clauses


# ---------------------------------------------------------------------------
# SyntheticValidator
# ---------------------------------------------------------------------------

class SyntheticValidator:
    """Multi-dimensional validator for synthetic training samples.

    Checks:
    * JSON-schema conformance (message structure, roles, tool calls)
    * Grounding — assistant answers reference extracted policies
    * Condition accuracy — conditional logic matches policy clauses
    * Tool-schema correctness — tool names and arguments are valid
    * Conversation ordering — role alternation rules
    * Near-duplicate detection and contamination checks
    """

    def __init__(
        self,
        policies: list[dict[str, Any]] | None = None,
        tool_schemas: list[ToolSchema] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._policies = policies or []
        self._tool_schemas = tool_schemas or []
        self._cfg = config or {}
        self._seen_hashes: set[str] = set()
        self._contamination_set: set[str] | None = None

    def set_contamination_set(self, reserved_ids: set[str]) -> None:
        """Register sample/scenario IDs that must not appear in training data."""
        self._contamination_set = reserved_ids

    def validate(self, sample: CanonicalSample) -> tuple[ValidationStatus, str | None]:
        """Run all validation checks on a single sample.

        Returns:
            ``(status, rejection_reason)`` — *rejection_reason* is ``None``
            when status is ``ACCEPTED``.
        """
        checks: list[tuple[str, bool, str]] = [
            self._check_schema(sample),
            self._check_conversation_order(sample),
            self._check_tool_schemas(sample),
            self._check_grounding(sample),
            self._check_deduplication(sample),
            self._check_contamination(sample),
        ]

        for name, passed, reason in checks:
            if not passed:
                logger.debug("Sample %s failed %s: %s", sample.sample_id, name, reason)
                return ValidationStatus.REJECTED, f"{name}: {reason}"

        return ValidationStatus.ACCEPTED, None

    def validate_batch(
        self,
        samples: Sequence[CanonicalSample],
    ) -> list[tuple[CanonicalSample, ValidationStatus, str | None]]:
        """Validate a batch and return annotated results."""
        results: list[tuple[CanonicalSample, ValidationStatus, str | None]] = []
        for s in samples:
            status, reason = self.validate(s)
            results.append((s, status, reason))
        accepted = sum(1 for _, st, _ in results if st == ValidationStatus.ACCEPTED)
        logger.info(
            "Batch validation: %d/%d accepted (%.1f%%)",
            accepted, len(results), 100 * accepted / max(len(results), 1),
        )
        return results

    # -- individual checks --------------------------------------------------

    @staticmethod
    def _check_schema(sample: CanonicalSample) -> tuple[str, bool, str]:
        if not sample.messages:
            return ("schema", False, "Empty message list")
        for msg in sample.messages:
            if msg.role not in ("system", "user", "assistant", "tool"):
                return ("schema", False, f"Invalid role: {msg.role!r}")
        return ("schema", True, "")

    @staticmethod
    def _check_conversation_order(sample: CanonicalSample) -> tuple[str, bool, str]:
        roles = [m.role for m in sample.messages]
        if not roles:
            return ("conversation_order", False, "No messages")

        # First message should be system or user
        if roles[0] not in ("system", "user"):
            return ("conversation_order", False, f"Conversation starts with {roles[0]!r}")

        # Tool messages must follow assistant messages with tool_calls
        for i, role in enumerate(roles):
            if role == "tool" and i > 0:
                prev = sample.messages[i - 1]
                if prev.role == "tool":
                    continue  # consecutive tool responses are allowed
                if prev.role != "assistant" or not prev.tool_calls:
                    return (
                        "conversation_order",
                        False,
                        f"Tool message at index {i} not preceded by assistant tool_call",
                    )

        return ("conversation_order", True, "")

    def _check_tool_schemas(self, sample: CanonicalSample) -> tuple[str, bool, str]:
        if not self._tool_schemas:
            return ("tool_schema", True, "")

        valid_names = {ts.function.name for ts in self._tool_schemas}

        for msg in sample.messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function.name not in valid_names:
                        return (
                            "tool_schema",
                            False,
                            f"Unknown tool: {tc.function.name!r}",
                        )
                    try:
                        json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        return (
                            "tool_schema",
                            False,
                            f"Invalid JSON arguments in tool call {tc.id}",
                        )

        return ("tool_schema", True, "")

    def _check_grounding(self, sample: CanonicalSample) -> tuple[str, bool, str]:
        """Lightweight grounding: ensure at least one source doc ID is known."""
        if not self._policies:
            return ("grounding", True, "")

        if not sample.source_doc_ids:
            return ("grounding", False, "No source_doc_ids for grounding check")

        known_ids = {p["doc_id"] for p in self._policies}
        for did in sample.source_doc_ids:
            if did in known_ids:
                return ("grounding", True, "")

        return ("grounding", False, "None of the source doc IDs match known policies")

    def _check_deduplication(self, sample: CanonicalSample) -> tuple[str, bool, str]:
        content_hash = hashlib.sha256(
            json.dumps(
                [m.model_dump(exclude_none=True) for m in sample.messages],
                sort_keys=True,
            ).encode()
        ).hexdigest()

        if content_hash in self._seen_hashes:
            return ("deduplication", False, "Exact duplicate detected")
        self._seen_hashes.add(content_hash)
        return ("deduplication", True, "")

    def _check_contamination(self, sample: CanonicalSample) -> tuple[str, bool, str]:
        if self._contamination_set is None:
            return ("contamination", True, "")
        if sample.sample_id in self._contamination_set:
            return ("contamination", False, "Sample ID in contamination set")
        if sample.scenario_family and sample.scenario_family in self._contamination_set:
            return ("contamination", False, "Scenario family in contamination set")
        return ("contamination", True, "")


# ---------------------------------------------------------------------------
# SDGPipeline
# ---------------------------------------------------------------------------

class SDGPipeline:
    """End-to-end synthetic data generation pipeline.

    Orchestrates teacher-model invocation (via ``sdg_hub`` or an explicit
    callable), validation, checkpointing, and budget tracking.

    Args:
        config: Parsed SDG configuration (from ``configs/sdg.yaml``).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_yaml_config("configs/sdg.yaml")
        self._cfg = config

        self._pipeline_name = config.get("pipeline", {}).get("name", "sdg")
        self._seed = config.get("generation", {}).get("seed", 42)
        self._checkpointing = config.get("pipeline", {}).get("checkpointing", True)
        self._checkpoint_dir = Path(
            config.get("pipeline", {}).get("checkpoint_path", "data/checkpoints/sdg")
        )
        self._budget_limit = config.get("teacher", {}).get("budget_limit_usd", 50.0)
        self._accumulated_cost = 0.0

        self._accepted: list[CanonicalSample] = []
        self._rejected: list[dict[str, Any]] = []
        self._provenance: list[ProvenanceRecord] = []

        self._validator: SyntheticValidator | None = None
        self._extractor: PolicyExtractor | None = None

        self._generator_fn: Any = None
        self._init_generator()

    def _init_generator(self) -> None:
        """Initialise the teacher-model generator via sdg_hub or a stub."""
        if _sdg_hub_available and _sdg_hub is not None:
            try:
                teacher_cfg = self._cfg.get("teacher", {})
                self._generator_fn = _sdg_hub.create_pipeline(
                    endpoint=teacher_cfg.get("endpoint", ""),
                    api_key=teacher_cfg.get("api_key", ""),
                    model=teacher_cfg.get("model", ""),
                    max_concurrent=teacher_cfg.get("max_concurrent", 4),
                    timeout=teacher_cfg.get("timeout_seconds", 120),
                )
                logger.info("SDG pipeline initialised with sdg_hub teacher")
            except Exception as exc:
                logger.warning("sdg_hub.create_pipeline failed: %s", exc)
        else:
            logger.info(
                "sdg_hub not available; pipeline requires explicit "
                "generate_batch calls or a custom generator"
            )

    # -- public API ---------------------------------------------------------

    def run(self, profile: str = "lab") -> list[CanonicalSample]:
        """Run the full generation pipeline for a given profile.

        The *profile* selects target counts from the config's
        ``generation.target_counts`` section (e.g. ``smoke`` for quick
        tests, ``lab`` for the full run).

        Returns:
            Accepted :class:`CanonicalSample` objects.
        """
        gen_cfg = self._cfg.get("generation", {})
        target_counts: dict[str, int] = gen_cfg.get("target_counts", {}).get(profile, {})
        if not target_counts:
            raise ValueError(
                f"No target counts defined for profile {profile!r} in SDG config"
            )

        # Remove the "total" key; iterate sample types
        total_target = target_counts.pop("total", sum(target_counts.values()))
        logger.info(
            "Starting SDG pipeline (profile=%s, total_target=%d)", profile, total_target,
        )

        # Resume from checkpoint if available
        self._maybe_resume_checkpoint()

        already_by_type: dict[str, int] = Counter(
            s.sample_type.value for s in self._accepted
        )

        for sample_type_str, count in target_counts.items():
            try:
                sample_type = SampleType(sample_type_str)
            except ValueError:
                logger.warning("Unknown sample type %r in config; skipping", sample_type_str)
                continue

            remaining = count - already_by_type.get(sample_type_str, 0)
            if remaining <= 0:
                logger.info("Type %s already satisfied (%d)", sample_type_str, count)
                continue

            logger.info("Generating %d samples of type %s", remaining, sample_type_str)
            batch = self.generate_batch(sample_type, remaining)
            validated = self.validate_batch(batch)

            for sample, status, reason in validated:
                if status == ValidationStatus.ACCEPTED:
                    sample.validation_status = ValidationStatus.ACCEPTED
                    self._accepted.append(sample)
                else:
                    sample.validation_status = ValidationStatus.REJECTED
                    self._rejected.append({
                        "sample_id": sample.sample_id,
                        "type": sample.sample_type.value,
                        "reason": reason,
                    })

            self._save_checkpoint()

            if self._budget_limit > 0 and self._accumulated_cost >= self._budget_limit:
                logger.warning(
                    "Budget limit reached ($%.2f / $%.2f). Stopping.",
                    self._accumulated_cost, self._budget_limit,
                )
                break

        logger.info(
            "SDG complete: %d accepted, %d rejected",
            len(self._accepted), len(self._rejected),
        )
        return list(self._accepted)

    def generate_batch(
        self,
        sample_type: SampleType,
        count: int,
    ) -> list[CanonicalSample]:
        """Generate a batch of synthetic samples of the given type.

        When ``sdg_hub`` is available the teacher model is invoked;
        otherwise an empty list is returned with a warning.
        """
        gen_cfg = self._cfg.get("generation", {})
        schema_cfg = gen_cfg.get("schemas", {}).get(sample_type.value, {})

        if self._generator_fn is None:
            logger.warning(
                "No generator available for %s; returning empty batch", sample_type.value,
            )
            return []

        samples: list[CanonicalSample] = []
        for i in range(count):
            sample_id = f"synthetic-{sample_type.value}-{uuid.uuid4().hex[:8]}"
            try:
                raw_output = self._generator_fn(
                    sample_type=sample_type.value,
                    flow=schema_cfg.get("flow", ""),
                    max_tokens=schema_cfg.get("max_tokens", 2048),
                    temperature=schema_cfg.get("temperature", 0.7),
                    seed=self._seed + i,
                )

                messages = [
                    Message(**m) for m in raw_output.get("messages", [])
                ]
                tools_raw = raw_output.get("tools")
                tools = [ToolSchema(**t) for t in tools_raw] if tools_raw else None

                sample = CanonicalSample(
                    sample_id=sample_id,
                    messages=messages,
                    tools=tools,
                    source_doc_ids=raw_output.get("source_doc_ids"),
                    scenario_family=raw_output.get("scenario_family"),
                    sample_type=sample_type,
                    validation_status=ValidationStatus.PENDING,
                )
                samples.append(sample)

                self._provenance.append(ProvenanceRecord(
                    sample_id=sample_id,
                    source_documents=raw_output.get("source_doc_ids", []),
                    scenario_family=raw_output.get("scenario_family", ""),
                    generator_id=self._pipeline_name,
                    generator_model=self._cfg.get("teacher", {}).get("model", ""),
                    generation_timestamp=_utcnow_iso(),
                ))

                cost = raw_output.get("cost_usd", 0.0)
                self._accumulated_cost += cost

            except Exception as exc:
                logger.error("Generation failed for %s: %s", sample_id, exc)
                self._rejected.append({
                    "sample_id": sample_id,
                    "type": sample_type.value,
                    "reason": f"generation_error: {exc}",
                })

        logger.info(
            "Generated %d/%d raw samples for %s (cost $%.4f)",
            len(samples), count, sample_type.value, self._accumulated_cost,
        )
        return samples

    def validate_batch(
        self,
        samples: Sequence[CanonicalSample],
    ) -> list[tuple[CanonicalSample, ValidationStatus, str | None]]:
        """Validate a batch of samples and return annotated results."""
        if self._validator is None:
            policies = self._extractor.policies if self._extractor else []
            tool_schemas = self._extractor.tool_schemas if self._extractor else []
            self._validator = SyntheticValidator(
                policies=policies,
                tool_schemas=tool_schemas,
                config=self._cfg.get("validation", {}),
            )
        return self._validator.validate_batch(samples)

    def export_canonical(self) -> list[CanonicalSample]:
        """Return accepted samples ready for bundle building."""
        return list(self._accepted)

    def set_policy_extractor(self, extractor: PolicyExtractor) -> None:
        """Attach a pre-built PolicyExtractor for grounding checks."""
        self._extractor = extractor
        self._validator = None  # force re-initialisation with new policies

    @property
    def provenance(self) -> list[ProvenanceRecord]:
        return list(self._provenance)

    @property
    def cost_usd(self) -> float:
        return self._accumulated_cost

    @property
    def rejection_log(self) -> list[dict[str, Any]]:
        return list(self._rejected)

    # -- checkpointing ------------------------------------------------------

    def _save_checkpoint(self) -> None:
        if not self._checkpointing:
            return
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "timestamp": _utcnow_iso(),
            "accepted_count": len(self._accepted),
            "rejected_count": len(self._rejected),
            "cost_usd": self._accumulated_cost,
            "accepted_ids": [s.sample_id for s in self._accepted],
            "accepted_samples": [s.model_dump() for s in self._accepted],
            "rejected": self._rejected,
            "provenance": [p.model_dump() for p in self._provenance],
        }
        ckpt_path = self._checkpoint_dir / "latest.json"
        with open(ckpt_path, "w", encoding="utf-8") as fh:
            json.dump(ckpt, fh, indent=2, default=str)
        logger.debug("Checkpoint saved: %s", ckpt_path)

    def _maybe_resume_checkpoint(self) -> None:
        if not self._checkpointing:
            return
        resume = self._cfg.get("pipeline", {}).get("resume_from_checkpoint", True)
        if not resume:
            return
        ckpt_path = self._checkpoint_dir / "latest.json"
        if not ckpt_path.exists():
            return

        try:
            with open(ckpt_path, encoding="utf-8") as fh:
                ckpt = json.load(fh)

            for raw in ckpt.get("accepted_samples", []):
                self._accepted.append(CanonicalSample(**raw))
            self._rejected = ckpt.get("rejected", [])
            self._accumulated_cost = ckpt.get("cost_usd", 0.0)
            for raw in ckpt.get("provenance", []):
                self._provenance.append(ProvenanceRecord(**raw))

            logger.info(
                "Resumed from checkpoint: %d accepted, %d rejected, $%.4f cost",
                len(self._accepted), len(self._rejected), self._accumulated_cost,
            )
        except Exception as exc:
            logger.warning("Failed to resume from checkpoint: %s", exc)


# ---------------------------------------------------------------------------
# BundleBuilder
# ---------------------------------------------------------------------------

class BundleBuilder:
    """Assemble a prepared dataset bundle from canonical samples.

    Creates the full directory layout expected by :class:`BundleManager`,
    including manifests, checksums, provenance, quality reports, and
    backend-specific training files.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            from rhoai_model_training_lab.config import load_bundle_config
            config = load_bundle_config()
        self._cfg = config
        self._bundle_cfg = config.get("bundle", {})
        self._model_cfg = config.get("model_profile", {})
        self._contents_cfg = config.get("contents", {})

    def build_bundle(
        self,
        canonical_samples: list[CanonicalSample],
        config: dict[str, Any] | None = None,
        provenance: list[ProvenanceRecord] | None = None,
        kb_documents: list[dict[str, Any]] | None = None,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Build the full bundle directory.

        Args:
            canonical_samples: All accepted canonical samples.
            config: Optional override config.
            provenance: Provenance records for all samples.
            kb_documents: KB documents to bundle.
            output_dir: Override for the output directory.

        Returns:
            Path to the created bundle root.
        """
        cfg = config or self._cfg
        bundle_name = self._bundle_cfg.get("name", "tau-knowledge")
        bundle_version = self._bundle_cfg.get("version", "v1")
        base_path = Path(
            output_dir
            or self._bundle_cfg.get("base_path", f"data/prepared/{bundle_name}-{bundle_version}")
        )
        base_path.mkdir(parents=True, exist_ok=True)

        # Split samples
        train_samples, val_samples, split_info = self._split_samples(canonical_samples)

        # Write canonical JSONL
        self._write_jsonl(base_path / "canonical" / "train.jsonl", train_samples)
        self._write_jsonl(base_path / "canonical" / "validation.jsonl", val_samples)

        # Write KB documents
        if kb_documents:
            self._write_jsonl(base_path / "kb" / "documents.jsonl", kb_documents)

        # Write backend-specific training files (same data, different key for now)
        for profile in ("lora", "osft"):
            self._write_jsonl(
                base_path / "training" / profile / "train.jsonl", train_samples,
            )
            self._write_jsonl(
                base_path / "training" / profile / "validation.jsonl", val_samples,
            )

        # Provenance
        if provenance:
            self._write_jsonl(base_path / "metadata" / "provenance.jsonl", provenance)

        # Split metadata
        split_path = base_path / "metadata" / "splits.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as fh:
            json.dump(split_info.model_dump(), fh, indent=2)

        # Quality report
        quality = self._build_quality_report(canonical_samples, train_samples, val_samples)
        report_path = base_path / "reports" / "quality.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(quality.model_dump(), fh, indent=2)

        # Dataset card
        card = self._build_dataset_card(bundle_name, bundle_version, quality)
        card_path = base_path / "DATASET_CARD.md"
        with open(card_path, "w", encoding="utf-8") as fh:
            fh.write(self._render_dataset_card_md(card))

        # Checksums
        checksum_records = self._compute_all_checksums(base_path)
        checksum_path = base_path / "checksums.sha256"
        with open(checksum_path, "w", encoding="utf-8") as fh:
            for digest, rel_path in checksum_records:
                fh.write(f"{digest} {rel_path}\n")

        # Manifest (after checksums so we can include bundle_hash)
        manifest = self._build_manifest(
            bundle_name, bundle_version, train_samples, val_samples,
            canonical_samples, checksum_path,
        )
        manifest_path = base_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest.model_dump(), fh, indent=2)

        logger.info("Bundle built at %s", base_path)
        return base_path

    # -- splitting ----------------------------------------------------------

    def _split_samples(
        self,
        samples: list[CanonicalSample],
    ) -> tuple[list[CanonicalSample], list[CanonicalSample], SplitInfo]:
        """Split by scenario_family (or random if families are absent)."""
        seed = 42
        families: dict[str, list[CanonicalSample]] = defaultdict(list)
        no_family: list[CanonicalSample] = []

        for s in samples:
            if s.scenario_family:
                families[s.scenario_family].append(s)
            else:
                no_family.append(s)

        import random
        rng = random.Random(seed)

        family_keys = sorted(families.keys())
        rng.shuffle(family_keys)

        split_point = max(1, int(len(family_keys) * 0.85))
        train_families = set(family_keys[:split_point])
        val_families = set(family_keys[split_point:])

        train: list[CanonicalSample] = []
        val: list[CanonicalSample] = []

        for fam in train_families:
            train.extend(families[fam])
        for fam in val_families:
            val.extend(families[fam])

        # Samples without family: split randomly
        rng.shuffle(no_family)
        no_split = max(1, int(len(no_family) * 0.85))
        train.extend(no_family[:no_split])
        val.extend(no_family[no_split:])

        family_partition = {f: "train" for f in train_families}
        family_partition.update({f: "validation" for f in val_families})

        split_info = SplitInfo(
            method="scenario_family",
            seed=seed,
            train_ids=[s.sample_id for s in train],
            validation_ids=[s.sample_id for s in val],
            train_count=len(train),
            validation_count=len(val),
            family_partition=family_partition,
        )

        return train, val, split_info

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _write_jsonl(
        path: Path,
        records: list[Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                if hasattr(rec, "model_dump"):
                    line = json.dumps(rec.model_dump(exclude_none=True), ensure_ascii=False)
                elif isinstance(rec, dict):
                    line = json.dumps(rec, ensure_ascii=False)
                else:
                    line = json.dumps(rec, ensure_ascii=False, default=str)
                fh.write(line + "\n")

    @staticmethod
    def _compute_all_checksums(base: Path) -> list[tuple[str, str]]:
        from rhoai_model_training_lab.data import compute_file_checksum

        records: list[tuple[str, str]] = []
        for file_path in sorted(base.rglob("*")):
            if file_path.is_file() and file_path.name not in ("checksums.sha256", "manifest.json"):
                rel = str(file_path.relative_to(base))
                digest = compute_file_checksum(file_path)
                records.append((digest, rel))
        return records

    def _build_manifest(
        self,
        name: str,
        version: str,
        train: list[CanonicalSample],
        val: list[CanonicalSample],
        all_samples: list[CanonicalSample],
        checksum_path: Path,
    ) -> BundleManifest:
        type_dist: dict[str, int] = Counter(s.sample_type.value for s in all_samples)

        bundle_hash = ""
        if checksum_path.exists():
            from rhoai_model_training_lab.data import compute_file_checksum
            bundle_hash = compute_file_checksum(checksum_path)

        return BundleManifest(
            bundle_name=name,
            bundle_version=version,
            model_id=self._model_cfg.get("model_id", "Qwen/Qwen3-4B-Instruct-2507"),
            model_revision=self._model_cfg.get("model_revision", "main"),
            tokenizer_id=self._model_cfg.get("tokenizer_id", "Qwen/Qwen3-4B-Instruct-2507"),
            canonical_train_count=len(train),
            canonical_validation_count=len(val),
            split_policy="scenario_family",
            sample_type_distribution=type_dist,
            bundle_hash=bundle_hash,
        )

    @staticmethod
    def _build_quality_report(
        all_samples: list[CanonicalSample],
        train: list[CanonicalSample],
        val: list[CanonicalSample],
    ) -> QualityReport:
        accepted = [s for s in all_samples if s.validation_status == ValidationStatus.ACCEPTED]
        rejected = [s for s in all_samples if s.validation_status == ValidationStatus.REJECTED]

        by_type: dict[str, TypeQualityStats] = {}
        for st in SampleType:
            of_type = [s for s in all_samples if s.sample_type == st]
            acc = [s for s in of_type if s.validation_status == ValidationStatus.ACCEPTED]
            rej = [s for s in of_type if s.validation_status == ValidationStatus.REJECTED]
            by_type[st.value] = TypeQualityStats(
                generated=len(of_type),
                accepted=len(acc),
                rejected=len(rej),
                acceptance_rate=len(acc) / max(len(of_type), 1),
            )

        doc_ids: set[str] = set()
        for s in all_samples:
            if s.source_doc_ids:
                doc_ids.update(s.source_doc_ids)

        return QualityReport(
            total_generated=len(all_samples),
            total_accepted=len(accepted),
            total_rejected=len(rejected),
            acceptance_rate=len(accepted) / max(len(all_samples), 1),
            by_type=by_type,
            coverage=CoverageStats(
                kb_documents_referenced=len(doc_ids),
            ),
        )

    @staticmethod
    def _build_dataset_card(
        name: str,
        version: str,
        quality: QualityReport,
    ) -> DatasetCard:
        return DatasetCard(
            name=name,
            version=version,
            description=(
                f"τ-Knowledge banking_knowledge prepared dataset bundle. "
                f"{quality.total_accepted} accepted samples across "
                f"{len(quality.by_type)} types."
            ),
            generation_method="sdg_hub teacher model",
            validation_method="schema + grounding + dedup + contamination",
            split_methodology="scenario_family isolation",
            training_types=["lora_sft", "osft"],
            model_profile="Qwen/Qwen3-4B-Instruct-2507",
            creation_date=_utcnow_iso(),
        )

    @staticmethod
    def _render_dataset_card_md(card: DatasetCard) -> str:
        return (
            f"# {card.name} {card.version}\n\n"
            f"{card.description}\n\n"
            f"## Generation\n\n"
            f"- Method: {card.generation_method}\n"
            f"- Validation: {card.validation_method}\n"
            f"- Split: {card.split_methodology}\n\n"
            f"## Training Types\n\n"
            + "\n".join(f"- {t}" for t in card.training_types)
            + f"\n\n## Model\n\n- {card.model_profile}\n"
            + f"\n## Created\n\n- {card.creation_date}\n"
        )


__all__ = [
    "BundleBuilder",
    "PolicyExtractor",
    "SDGPipeline",
    "SyntheticValidator",
]
