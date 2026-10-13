"""Regression checks for selected-body HTML and decode-attempt accounting."""

import sys
import unittest
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spam_classifier.email_parser import clean_text, parse_email


class ParserStatisticsTests(unittest.TestCase):
    def test_simple_ascii_decoding_character_accounting(self):
        result = parse_email(b'Subject: Hi\r\nContent-Type: text/plain; charset=us-ascii\r\n\r\nA body')
        self.assertEqual(result["decoded_characters"], len("Hi") + len("A body"))
        self.assertEqual(result["replacement_characters"], 0)
        self.assertEqual(result["replacement_ratio"], 0)

    def test_contact_masking_is_idempotent_after_space_normalization(self):
        first = clean_text("Please call 555   123   4567 now")
        self.assertNotIn("555", first)
        self.assertEqual(first, clean_text(first))

    def test_unselected_html_is_not_reported_as_html_body(self):
        message = EmailMessage()
        message.set_content("normal plain body")
        message.add_alternative("<p>alternative HTML body</p>", subtype="html")
        result = parse_email(message.as_bytes())
        self.assertFalse(result["has_html"])
        self.assertTrue(result["has_html_part_anywhere"])
        self.assertIn("normal plain body", result["body_clean"])
        self.assertNotIn("alternative html", result["body_clean"])

    def test_html_attachment_is_not_reported_as_html_body(self):
        message = EmailMessage()
        message.set_content("normal plain body")
        message.add_attachment("<p>private attachment</p>", subtype="html", filename="private.html")
        result = parse_email(message.as_bytes())
        self.assertFalse(result["has_html"])
        self.assertTrue(result["has_html_part_anywhere"])
        self.assertNotIn("private attachment", result["body_clean"])

    def test_rejected_candidate_counts_only_in_attempt_statistics(self):
        raw = (b'Content-Type: multipart/alternative; boundary="x"\r\n\r\n'
               b'--x\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n\xff\r\n'
               b'--x\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<p>valid body</p>\r\n--x--\r\n')
        result = parse_email(raw)
        self.assertTrue(result["usable"])
        self.assertIn("valid body", result["body_clean"])
        self.assertGreater(result["replacement_characters"], 0)
        self.assertEqual(result["retained_replacement_characters"], 0)
        self.assertEqual(result["retained_characters"], len(result["text_clean"]))

    def test_duplicate_mixed_representations_do_not_repeat_body_text(self):
        message = EmailMessage()
        message.make_mixed()
        for _ in range(2):
            part = EmailMessage()
            part.set_content("same readable text")
            message.attach(part)
        result = parse_email(message.as_bytes())
        self.assertEqual(result["body_clean"].count("same readable text"), 1)
        self.assertIn("duplicate_body_representation", result["recovery_actions"])


if __name__ == "__main__":
    unittest.main()
