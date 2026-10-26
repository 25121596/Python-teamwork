"""Tests for the guide and deck consistency check.

The check is a guard against the narrative drifting away from the published
evidence, so the tests cover both the happy path and the failure it exists
to catch: a figure that no longer appears in the guide or the deck.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "check_guide_consistency.py"
    spec = importlib.util.spec_from_file_location("check_guide_consistency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return load_module()


def test_facts_are_derived_from_published_aggregates(checker):
    facts = checker.expected_facts()
    assert facts["total_mail"] == "64,620"
    assert facts["representatives"] == "33,988"
    assert facts["train_rows"] == "23,792"
    assert facts["validation_rows"] == "5,099"
    assert facts["test_rows"] == "5,097"
    assert facts["lr_accuracy"] == "99.22%"
    assert facts["lr_macro_f1"] == "99.16%"
    assert facts["lr_fp"] == "11"
    assert facts["lr_fn"] == "29"


def test_guide_and_deck_match_the_published_evidence(checker):
    report = checker.run()
    assert report["missing_in_guide"] == []
    assert report["missing_in_deck"] == []
    assert report["passed"] is True
    assert report["checked_facts"] >= 20


def test_guide_is_required_to_carry_every_headline_figure(checker):
    facts = checker.expected_facts()
    guide = checker.GUIDE.read_text(encoding="utf-8")
    missing = checker.missing_from(guide, facts)
    assert missing == [], f"the guide stopped quoting: {missing}"


def test_missing_figure_is_reported(checker):
    facts = {"lr_accuracy": "99.22%"}
    assert checker.missing_from("nothing relevant here", facts) == ["lr_accuracy"]
    assert checker.missing_from("accuracy is 99.22% overall", facts) == []


def test_deck_omissions_are_declared_and_bounded(checker):
    for key in checker.DECK_OMITTED:
        assert key in checker.expected_facts()
    assert len(checker.DECK_OMITTED) < 5


def test_deck_text_reads_tables_and_text_frames(checker):
    text = checker.deck_text()
    assert "99.22%" in text
    assert "64,620" in text
    assert len(text) > 1000
