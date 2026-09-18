"""Tests for split isolation: scenario_family, canonical IDs, and holdout disjointness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rhoai_model_training_lab.schemas.data import SplitInfo


class TestScenarioFamilyIsolation:
    """Test that scenario_family members stay in the same split."""

    def test_family_members_same_split(self, sample_bundle_path):
        splits_path = sample_bundle_path / "metadata" / "splits.json"
        split_info = SplitInfo(**json.loads(splits_path.read_text()))

        for family, assigned_split in split_info.family_partition.items():
            assert assigned_split in ("train", "validation"), (
                f"Family {family!r} assigned to unexpected split {assigned_split!r}"
            )

    def test_no_family_in_both_splits(self, sample_bundle_path):
        splits_path = sample_bundle_path / "metadata" / "splits.json"
        split_info = SplitInfo(**json.loads(splits_path.read_text()))

        train_families = {
            f for f, s in split_info.family_partition.items() if s == "train"
        }
        val_families = {
            f for f, s in split_info.family_partition.items() if s == "validation"
        }
        overlap = train_families & val_families
        assert overlap == set(), f"Families appear in both splits: {overlap}"


class TestCanonicalIDDisjointness:
    """Test no canonical ID appears in both train and validation."""

    def test_no_id_in_both_splits(self, sample_bundle_path):
        splits_path = sample_bundle_path / "metadata" / "splits.json"
        split_info = SplitInfo(**json.loads(splits_path.read_text()))

        train_ids = set(split_info.train_ids)
        val_ids = set(split_info.validation_ids)
        overlap = train_ids & val_ids
        assert overlap == set(), f"IDs appear in both train and validation: {overlap}"

    def test_all_ids_accounted_for(self, sample_bundle_path):
        splits_path = sample_bundle_path / "metadata" / "splits.json"
        split_info = SplitInfo(**json.loads(splits_path.read_text()))

        total = len(split_info.train_ids) + len(split_info.validation_ids)
        expected = split_info.train_count + split_info.validation_count
        assert total == expected, f"ID count mismatch: {total} != {expected}"


class TestEvaluationHoldoutDisjoint:
    """Test evaluation holdout IDs are disjoint from training."""

    def test_holdout_disjoint_from_train(self, sample_bundle_path):
        splits_path = sample_bundle_path / "metadata" / "splits.json"
        split_info = SplitInfo(**json.loads(splits_path.read_text()))

        train_ids = set(split_info.train_ids)
        holdout_ids = set(split_info.holdout_ids)

        overlap = train_ids & holdout_ids
        assert overlap == set(), f"Holdout IDs leak into training: {overlap}"

    def test_holdout_disjoint_from_validation(self, sample_bundle_path):
        splits_path = sample_bundle_path / "metadata" / "splits.json"
        split_info = SplitInfo(**json.loads(splits_path.read_text()))

        val_ids = set(split_info.validation_ids)
        holdout_ids = set(split_info.holdout_ids)

        overlap = val_ids & holdout_ids
        assert overlap == set(), f"Holdout IDs leak into validation: {overlap}"


class TestMultiFamilySplit:
    """Test split isolation with multiple scenario families."""

    @pytest.fixture
    def multi_family_split(self) -> SplitInfo:
        return SplitInfo(
            method="scenario_family",
            seed=42,
            train_ids=["s1", "s2", "s3", "s4", "s5", "s6"],
            validation_ids=["s7", "s8"],
            holdout_ids=["s9", "s10"],
            train_count=6,
            validation_count=2,
            family_partition={
                "account_opening": "train",
                "fee_inquiry": "train",
                "loan_application": "validation",
            },
        )

    def test_families_assigned_to_single_split(self, multi_family_split):
        partitions = multi_family_split.family_partition
        train_fams = {f for f, s in partitions.items() if s == "train"}
        val_fams = {f for f, s in partitions.items() if s == "validation"}
        assert train_fams & val_fams == set()

    def test_train_val_holdout_all_disjoint(self, multi_family_split):
        s = multi_family_split
        train = set(s.train_ids)
        val = set(s.validation_ids)
        hold = set(s.holdout_ids)

        assert train & val == set()
        assert train & hold == set()
        assert val & hold == set()

    def test_total_ids_match_counts(self, multi_family_split):
        s = multi_family_split
        assert len(s.train_ids) == s.train_count
        assert len(s.validation_ids) == s.validation_count
