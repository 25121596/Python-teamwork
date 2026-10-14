"""Synthetic tests for content deduplication and group-safe data splitting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spam_classifier.data_split import assign_splits, prepare_records, signature


RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def row(
    mail_id: str,
    *,
    label: str = "ham",
    text: str | None = None,
    body: str | None = None,
    raw_hash: str | None = None,
    html_text: str | None = None,
    usable: bool = True,
    body_empty: bool | None = None,
) -> dict:
    text = text if text is not None else (f"text {mail_id}" if usable else "")
    body = body if body is not None else f"body {mail_id}"
    return {
        "mail_id": mail_id,
        "label_name": label,
        "label": {"ham": 0, "spam": 1}[label],
        "text_clean": text,
        "body_clean": body,
        "body_empty": not bool(body) if body_empty is None else body_empty,
        "usable": usable,
        "raw_sha256": raw_hash if raw_hash is not None else f"raw-{mail_id}",
        "text_unstripped_html": html_text if html_text is not None else f"html-{mail_id}",
    }


class DataSplitTests(unittest.TestCase):
    def prepared(self, rows: list[dict], development_ids: set[str] | None = None):
        return prepare_records(rows, development_ids or set())

    def ledger_by_id(self, preparation) -> dict[str, dict]:
        ledger = preparation.ledger
        self.assertEqual(len(ledger), len({item["mail_id"] for item in ledger}))
        return {item["mail_id"]: item for item in ledger}

    def test_signature_is_stable_nonempty_and_field_scoped(self):
        first = signature("Example content", "text_clean")
        self.assertIsInstance(first, str)
        self.assertTrue(first)
        self.assertEqual(first, signature("Example content", "text_clean"))
        self.assertNotEqual(first, signature("Other content", "text_clean"))
        self.assertNotEqual(first, signature("Example content", "body_clean"))

    def test_prepare_returns_expected_container_and_row_contract(self):
        result = self.prepared([row("one")])
        self.assertIsInstance(result.ledger, list)
        self.assertIsInstance(result.representatives, list)
        self.assertIsInstance(result.summary, dict)
        entry = result.ledger[0]
        self.assertEqual(entry["mail_id"], "one")
        for key in ("label_name", "label", "disposition", "representative_id", "leakage_group_id", "forced_train"):
            self.assertIn(key, entry)
        self.assertIn(entry["disposition"], {"retained", "duplicate", "conflict", "unusable"})
        self.assertIsInstance(entry["forced_train"], bool)

    def test_content_duplicates_keep_one_representative_and_dev_row_wins(self):
        rows = [
            row("a-lexically-first", text="Same normalized content", body="same body"),
            row("z-development", text="Same normalized content", body="same body"),
            row("third-copy", text="Same normalized content", body="same body"),
        ]
        result = self.prepared(rows, {"z-development"})
        ledger = self.ledger_by_id(result)
        self.assertEqual(len(result.representatives), 1)
        self.assertEqual(result.representatives[0]["mail_id"], "z-development")
        self.assertEqual(ledger["z-development"]["disposition"], "retained")
        self.assertTrue(ledger["z-development"]["forced_train"])
        for duplicate_id in ("a-lexically-first", "third-copy"):
            self.assertEqual(ledger[duplicate_id]["disposition"], "duplicate")
            self.assertEqual(ledger[duplicate_id]["representative_id"], "z-development")
            self.assertTrue(ledger[duplicate_id]["forced_train"])

    def test_multiple_development_duplicates_still_force_the_whole_group_to_train(self):
        rows = [
            row("dev-b", text="Repeated dev content", body="shared"),
            row("dev-a", text="Repeated dev content", body="shared"),
            row("outside", text="Repeated dev content", body="shared"),
        ]
        result = self.prepared(rows, {"dev-a", "dev-b"})
        ledger = self.ledger_by_id(result)
        self.assertEqual(len(result.representatives), 1)
        self.assertEqual(result.representatives[0]["mail_id"], "dev-a")
        self.assertTrue(all(entry["forced_train"] for entry in ledger.values()))

    def test_transitive_body_hash_text_and_html_links_form_one_leakage_group(self):
        rows = [
            row("a", text="topic a", body="shared body ab", raw_hash="raw-a", html_text="html-a"),
            row("b", text="topic b", body="shared body ab", raw_hash="shared raw bc", html_text="html-b"),
            row("c", text="topic c", body="body-c", raw_hash="shared raw bc", html_text="shared html cd"),
            row("d", text="topic d", body="body-d", raw_hash="raw-d", html_text="shared html cd"),
        ]
        result = self.prepared(rows)
        self.assertEqual(len(result.representatives), 4)
        group_ids = {entry["leakage_group_id"] for entry in result.ledger}
        self.assertEqual(len(group_ids), 1)

    def test_empty_bodies_do_not_create_edges_between_unrelated_records(self):
        rows = [
            row("empty-a", text="topic a", body="", html_text="html-a"),
            row("empty-b", text="topic b", body="", html_text="html-b"),
            row("symbolic-empty-a", text="topic c", body="---", body_empty=True, html_text="html-c"),
            row("symbolic-empty-b", text="topic d", body="___", body_empty=True, html_text="html-d"),
        ]
        result = self.prepared(rows)
        ledger = self.ledger_by_id(result)
        self.assertNotEqual(ledger["empty-a"]["leakage_group_id"], ledger["empty-b"]["leakage_group_id"])
        self.assertNotEqual(ledger["symbolic-empty-a"]["leakage_group_id"],
                            ledger["symbolic-empty-b"]["leakage_group_id"])

    def test_shared_body_across_labels_groups_rows_without_label_conflict(self):
        rows = [
            row("ham-topic", label="ham", text="ham topic", body="same quoted boilerplate"),
            row("spam-topic", label="spam", text="spam topic", body="same quoted boilerplate"),
        ]
        result = self.prepared(rows)
        ledger = self.ledger_by_id(result)
        self.assertEqual(len(result.representatives), 2)
        self.assertTrue(all(entry["disposition"] == "retained" for entry in ledger.values()))
        self.assertEqual(ledger["ham-topic"]["leakage_group_id"], ledger["spam-topic"]["leakage_group_id"])

    def test_cross_label_raw_hash_or_exact_text_conflicts_isolate_every_row_for_key(self):
        rows = [
            row("raw-ham", label="ham", text="raw key ham", raw_hash="same-raw-key"),
            row("raw-spam", label="spam", text="raw key spam", raw_hash="same-raw-key"),
            row("raw-ham-copy", label="ham", text="another row for raw key", raw_hash="same-raw-key"),
            row("text-ham", label="ham", text="same exact text key", raw_hash="hash-text-ham"),
            row("text-spam", label="spam", text="same exact text key", raw_hash="hash-text-spam"),
            row("text-ham-copy", label="ham", text="same exact text key", raw_hash="hash-text-copy"),
        ]
        result = self.prepared(rows)
        ledger = self.ledger_by_id(result)
        self.assertEqual(result.representatives, [])
        self.assertTrue(all(entry["disposition"] == "conflict" for entry in ledger.values()))

    def test_development_conflict_forces_its_connected_rows_to_train(self):
        rows = [
            row("dev-conflict", label="ham", text="different ham text", raw_hash="colliding-hash"),
            row("other-conflict", label="spam", text="different spam text", raw_hash="colliding-hash"),
        ]
        result = self.prepared(rows, {"dev-conflict"})
        ledger = self.ledger_by_id(result)
        self.assertTrue(all(entry["disposition"] == "conflict" for entry in ledger.values()))
        self.assertTrue(all(entry["forced_train"] for entry in ledger.values()))

    def test_unusable_rows_are_not_representatives(self):
        result = self.prepared([row("unusable", usable=False)])
        ledger = self.ledger_by_id(result)
        self.assertEqual(ledger["unusable"]["disposition"], "unusable")
        self.assertEqual(result.representatives, [])

    def test_split_assignment_is_reproducible_and_independent_of_input_order(self):
        rows = [row(f"mail-{index:03d}", label="ham" if index % 2 == 0 else "spam") for index in range(60)]
        reps = self.prepared(rows).representatives
        assigned = assign_splits(reps, seed=42, ratios=RATIOS)
        repeated = assign_splits(reps, seed=42, ratios=RATIOS)
        shuffled_order = assign_splits(list(reversed(reps)), seed=42, ratios=RATIOS)
        by_id = {entry["mail_id"]: entry["split"] for entry in assigned}
        self.assertEqual(by_id, {entry["mail_id"]: entry["split"] for entry in repeated})
        self.assertEqual(by_id, {entry["mail_id"]: entry["split"] for entry in shuffled_order})
        self.assertEqual(set(by_id.values()), {"train", "validation", "test"})

    def test_split_assignment_keeps_each_leakage_group_together_and_development_in_train(self):
        rows = [row(f"mail-{index:03d}") for index in range(40)]
        rows.extend([
            row("linked-dev", text="unique linked dev", body="linked component body"),
            row("linked-other", text="unique linked other", body="linked component body"),
        ])
        reps = self.prepared(rows, {"linked-dev"}).representatives
        assigned = assign_splits(reps, seed=42, ratios=RATIOS)
        split_by_group: dict[str, set[str]] = {}
        by_id = {entry["mail_id"]: entry for entry in assigned}
        for entry in assigned:
            split_by_group.setdefault(entry["leakage_group_id"], set()).add(entry["split"])
        self.assertTrue(all(len(splits) == 1 for splits in split_by_group.values()))
        linked_group = by_id["linked-dev"]["leakage_group_id"]
        self.assertEqual(by_id["linked-other"]["leakage_group_id"], linked_group)
        self.assertEqual(by_id["linked-dev"]["split"], "train")
        self.assertEqual(by_id["linked-other"]["split"], "train")

    def test_invalid_split_ratios_are_rejected(self):
        reps = self.prepared([row("one")]).representatives
        invalid = (
            {"train": 0.7, "validation": 0.2, "test": 0.2},
            {"train": 1.1, "validation": -0.05, "test": -0.05},
            {"train": 0.7, "validation": 0.3},
        )
        for ratios in invalid:
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                assign_splits(reps, seed=42, ratios=ratios)

    def test_duplicate_mail_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            self.prepared([row("same-id"), row("same-id", text="different text")])

    def test_development_ids_must_be_subset_of_input_ids(self):
        with self.assertRaises(ValueError):
            self.prepared([row("present")], {"absent"})

    def test_usable_flag_must_match_clean_text_signal(self):
        inconsistent = row("flag-mismatch", text="visible content", usable=False)
        with self.assertRaises(ValueError):
            self.prepared([inconsistent])


if __name__ == "__main__":
    unittest.main()
