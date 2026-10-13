"""Synthetic tests for MIME parsing; never reads corpus mail files."""

from __future__ import annotations

import sys
import unittest
from email.header import Header
from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spam_classifier.email_parser import clean_text, parse_email


class EmailParserTests(unittest.TestCase):
    fields = {
        "subject_clean", "body_clean", "text_clean", "body_decoded", "parse_status",
        "usable", "mime_defects", "recovery_actions", "decode_fallback",
        "replacement_ratio", "has_html", "skipped_attachments", "skipped_forwarded",
    }
    statuses = {"ok", "recovered", "subject_only", "empty", "quarantined", "error"}

    def parsed(self, raw: bytes) -> dict:
        result = parse_email(raw)
        self.assertIsInstance(result, dict)
        self.assertTrue(self.fields.issubset(result))
        self.assertIn(result["parse_status"], self.statuses)
        self.assertIsInstance(result["usable"], bool)
        self.assertIsInstance(result["mime_defects"], list)
        self.assertTrue(all(isinstance(x, str) for x in result["mime_defects"]))
        self.assertIsInstance(result["recovery_actions"], list)
        self.assertTrue(all(isinstance(x, str) for x in result["recovery_actions"]))
        self.assertIsInstance(result["decode_fallback"], bool)
        self.assertIsInstance(result["replacement_ratio"], (int, float))
        self.assertIsInstance(result["has_html"], bool)
        self.assertIsInstance(result["skipped_attachments"], int)
        self.assertIsInstance(result["skipped_forwarded"], int)
        for name in ("subject_clean", "body_clean", "text_clean", "body_decoded"):
            self.assertIsInstance(result[name], str)
        return result

    def test_rfc2047_gb2312_subject_and_body(self):
        subject = Header("中文主题", "gb2312").encode()
        raw = (f"Subject: {subject}\r\nContent-Type: text/plain; charset=gb2312\r\n\r\n".encode("ascii")
               + "中文正文内容".encode("gb2312"))
        result = self.parsed(raw)
        self.assertIn("中文主题", result["subject_clean"])
        self.assertIn("中文正文内容", result["text_clean"])

    def test_utf8_base64_and_quoted_printable(self):
        for cte in ("base64", "quoted-printable"):
            with self.subTest(cte=cte):
                message = EmailMessage(policy=SMTP)
                message["Subject"] = "encoded body"
                message.set_content("传输编码正文 marker", charset="utf-8", cte=cte)
                self.assertIn("传输编码正文 marker", self.parsed(message.as_bytes())["body_clean"])

    def test_html_strips_script_style_and_tags_keeps_entities_paragraphs_and_anchor(self):
        raw = (b"Content-Type: text/html; charset=utf-8\r\n\r\n"
               b"<style>private_style_marker</style><script>private_script_marker()</script>"
               b"<p>First &amp; safe</p><p><a href='https://example.test/x'>Visible link text</a></p>")
        result = self.parsed(raw)
        self.assertTrue(result["has_html"])
        for expected in ("first & safe", "visible link text"):
            self.assertIn(expected, result["body_clean"])
        for removed in ("private_style_marker", "private_script_marker", "<p>"):
            self.assertNotIn(removed, result["body_clean"])

    def test_alternative_prefers_only_valid_plain_part(self):
        msg = EmailMessage(policy=SMTP)
        msg.set_content("Plain alternative unique marker")
        msg.add_alternative("<p>HTML alternative unique marker</p>", subtype="html")
        result = self.parsed(msg.as_bytes())
        self.assertIn("plain alternative unique marker", result["body_clean"])
        self.assertNotIn("html alternative unique marker", result["body_clean"])
        self.assertEqual(result["body_clean"].count("plain alternative unique marker"), 1)

    def test_alternative_falls_back_to_html_when_plain_part_is_unusable(self):
        raw = (b"MIME-Version: 1.0\r\nContent-Type: multipart/alternative; boundary=a\r\n\r\n"
               b"--a\r\nContent-Type: text/plain; charset=ascii\r\nContent-Transfer-Encoding: base64\r\n\r\n/w==\r\n"
               b"--a\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<p>HTML fallback usable marker</p>\r\n--a--\r\n")
        self.assertIn("html fallback usable marker", self.parsed(raw)["body_clean"])

    def test_mixed_attachment_is_skipped(self):
        msg = EmailMessage(policy=SMTP)
        msg.set_content("Visible body only")
        msg.add_attachment(b"attachment_private_marker", maintype="application", subtype="octet-stream",
                           filename="data.bin")
        result = self.parsed(msg.as_bytes())
        self.assertIn("visible body only", result["body_clean"])
        self.assertNotIn("attachment_private_marker", result["text_clean"])
        self.assertGreaterEqual(result["skipped_attachments"], 1)

    def test_forwarded_rfc822_message_is_skipped(self):
        outer = EmailMessage(policy=SMTP)
        outer.set_content("Outer body only")
        inner = EmailMessage(policy=SMTP)
        inner["Subject"] = "forwarded private subject"
        inner.set_content("forwarded_private_body_marker")
        outer.add_attachment(inner)
        result = self.parsed(outer.as_bytes())
        self.assertIn("outer body only", result["body_clean"])
        self.assertNotIn("forwarded_private_body_marker", result["text_clean"])
        self.assertGreaterEqual(result["skipped_forwarded"], 1)

    def test_unknown_charset_falls_back(self):
        raw = b"Content-Type: text/plain; charset=x-unknown-test-charset\r\n\r\n" + "可读回退文本".encode()
        result = self.parsed(raw)
        self.assertIn("可读回退文本", result["body_clean"])
        self.assertTrue(result["decode_fallback"])

    def test_gbk_body_mislabeled_latin1_is_recovered(self):
        chinese = "这是一段足够长的中文内容用于检测错误字符集标记是否能够恢复正常编码"
        raw = (b"Content-Type: text/plain; charset=latin1\r\n\r\n" + chinese.encode("gbk"))
        result = self.parsed(raw)
        self.assertIn(chinese, result["body_clean"])
        self.assertTrue(result["decode_fallback"] or result.get("encoding_warnings"))

    def test_unknown_transfer_encoding_does_not_hide_subject_or_html_alternative(self):
        raw = (b"Subject: Keep this subject\r\nMIME-Version: 1.0\r\n"
               b"Content-Type: multipart/alternative; boundary=u\r\n\r\n"
               b"--u\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: x-unknown\r\n\r\nopaque\r\n"
               b"--u\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<p>Usable alternative text</p>\r\n--u--\r\n")
        result = self.parsed(raw)
        self.assertIn("keep this subject", result["subject_clean"])
        self.assertIn("usable alternative text", result["body_clean"])

    def test_broken_base64_alternative_does_not_prevent_html_recovery(self):
        raw = (b"MIME-Version: 1.0\r\nContent-Type: multipart/alternative; boundary=b\r\n\r\n"
               b"--b\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: base64\r\n\r\n%%%invalid%%%\r\n"
               b"--b\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<p>Alternative after broken base64</p>\r\n--b--\r\n")
        result = self.parsed(raw)
        self.assertNotEqual(result["parse_status"], "error")
        self.assertIn("alternative after broken base64", result["body_clean"])

    def test_readable_raw_chinese_survives_bad_base64_declaration(self):
        raw = (b"Content-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: base64\r\n\r\n"
               + "解码前已经可读的中文正文内容".encode("utf-8"))
        result = self.parsed(raw)
        self.assertIn("解码前已经可读的中文正文内容", result["body_clean"])
        self.assertTrue(result["recovery_actions"] or result["parse_status"] == "recovered")

    def test_related_start_selects_nonfirst_html_root_and_ignores_plain_resource(self):
        raw = (b"MIME-Version: 1.0\r\nContent-Type: multipart/related; boundary=r; start=\"<root>\"\r\n\r\n"
               b"--r\r\nContent-Type: text/plain; charset=utf-8\r\nContent-ID: <resource>\r\n\r\nRESOURCE_PRIVATE_MARKER\r\n"
               b"--r\r\nContent-Type: text/html; charset=utf-8\r\nContent-ID: <root>\r\n\r\n<p>Selected HTML root</p>\r\n--r--\r\n")
        result = self.parsed(raw)
        self.assertIn("selected html root", result["body_clean"])
        self.assertNotIn("RESOURCE_PRIVATE_MARKER", result["text_clean"])

    def test_broken_multipart_plain_payload_recovers_and_reports_defect(self):
        raw = (b"MIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=missing\r\n\r\n"
               b"Recoverable plain text after broken declaration")
        result = self.parsed(raw)
        self.assertIn("recoverable plain text", result["text_clean"])
        self.assertTrue(result["mime_defects"])

    def test_broken_multipart_long_dash_decoration_does_not_hide_text(self):
        raw = (b"MIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=declared-boundary\r\n\r\n"
               b"Readable body preceding a decorative line\r\n----------------------\r\nReadable body after it")
        result = self.parsed(raw)
        self.assertIn("readable body preceding", result["text_clean"])
        self.assertIn("readable body after it", result["text_clean"])

    def test_empty_message(self):
        result = self.parsed(b"")
        self.assertEqual(result["parse_status"], "empty")
        self.assertFalse(result["usable"])
        self.assertEqual(result["body_clean"], "")

    def test_subject_only_message_is_usable(self):
        result = self.parsed(b"Subject: Useful subject\r\nFrom: sender@example.test\r\n\r\n")
        self.assertEqual(result["parse_status"], "subject_only")
        self.assertTrue(result["usable"])
        self.assertIn("useful subject", result["subject_clean"])

    def test_binary_payload_is_not_used_as_body_text(self):
        raw = b"Content-Type: image/jpeg\r\nContent-Transfer-Encoding: base64\r\n\r\n/9j/2Q=="
        result = self.parsed(raw)
        self.assertIn(result["parse_status"], {"quarantined", "empty"})
        self.assertEqual(result["body_clean"], "")
        self.assertFalse(result["usable"])

    def test_header_identity_is_not_in_cleaned_text(self):
        raw = (b"From: Sender Identity <sender@example.test>\r\nTo: Recipient Identity <to@example.test>\r\n"
               b"Subject: Ordinary subject\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nJust the body")
        result = self.parsed(raw)
        for identity in ("Sender Identity", "sender@example.test", "Recipient Identity", "to@example.test", "From:", "To:"):
            self.assertNotIn(identity, result["text_clean"])

    def test_contact_values_are_removed_or_replaced_by_clean_text(self):
        source = "Visit https://example.test/path now; mail person@example.test or call 555-123-4567."
        cleaned = clean_text(source)
        for original in ("https://example.test/path", "person@example.test", "555-123-4567"):
            self.assertNotIn(original, cleaned)

    def test_unicode_controls_are_removed_and_short_replacement_subject_survives(self):
        cleaned = clean_text("Keep\x00 this\x07 readable\u202e text\u200b")
        for control in ("\x00", "\x07", "\u202e", "\u200b"):
            self.assertNotIn(control, cleaned)
        subject = "短主题�仍保留"
        result = self.parsed((f"Subject: {subject}\r\n\r\n").encode("utf-8"))
        self.assertIn("短主题", result["subject_clean"])
        self.assertIn("仍保留", result["subject_clean"])


if __name__ == "__main__":
    unittest.main()
