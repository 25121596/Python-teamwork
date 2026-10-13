"""Decode and clean mail text; private text must stay in the local cache."""

from __future__ import annotations

import base64
import codecs
import copy
import quopri
import re
import unicodedata
from dataclasses import dataclass
from email import policy
from email.header import decode_header
from email.message import Message
from email.parser import BytesParser

from bs4 import BeautifulSoup, Comment


PARSER_VERSION = "1.0"
REPLACEMENT_LIMIT = 0.02
CONTROL_LIMIT = 0.02
HTML_TAG = re.compile(r"<(?:html|body|div|p|br|table|a|span|font|style|script|img|meta)\b", re.I)
MIME_HEADER_BLOCK = re.compile(r"(?im)^(?:content-(?:type|disposition|transfer-encoding)\s*:[^\n]*\n){2,}")
URL = re.compile(r"(?:https?://|www\.)[^\s<>\"'，。；！？（）]+", re.I)
EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)
PHONE = re.compile(
    r"(?<!\d)(?:\+?86[- ]?)?(?:1[3-9]\d{9}|0\d{2,3}[- ]?\d{7,8})(?!\d)"
    r"|(?<!\d)(?:\+\d{1,3}[- .]?)?\(?\d{2,4}\)?[- .]\d{3,4}[- .]\d{4}(?!\d)"
    r"|(?<!\d)\+\d{8,15}(?!\d)"
)


def clean_text(text: str) -> str:
    """Normalize Unicode and obvious contact identifiers without dropping digits."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf", "Cs"}
    )
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    text = URL.sub(" urltoken ", text)
    text = EMAIL.sub(" emailtoken ", text)
    text = PHONE.sub(" phonetoken ", text)
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for node in soup.find_all(["script", "style", "head", "template", "noscript"]):
        node.decompose()
    for node in soup.find_all(string=lambda value: isinstance(value, Comment)):
        node.extract()
    for node in soup.find_all("br"):
        node.replace_with("\n")
    for node in soup.find_all(["p", "div", "tr", "li", "h1", "h2", "h3", "section"]):
        node.append("\n")
    return soup.get_text(separator=" ", strip=False)


def _control_count(text: str) -> int:
    return sum(unicodedata.category(char) in {"Cc", "Cs"} and char not in "\n\r\t" for char in text)


@dataclass
class Decoded:
    text: str
    declared: str | None
    used: str
    fallback: bool
    warning: str = ""


def _cjk_count(text: str) -> int:
    return sum("\u3400" <= char <= "\u9fff" for char in text)


def _decode_bytes(data: bytes, declared: str | None = None) -> Decoded:
    if not data:
        return Decoded("", declared, "ascii", False)
    candidates = []
    if declared:
        try:
            candidates.append(codecs.lookup(declared.strip()).name)
        except (LookupError, ValueError):
            pass
    elif data.isascii():
        return Decoded(data.decode("ascii"), None, "ascii", False)
    hint_codecs = []
    for marker, codec in ((codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
                          (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"),
                          (codecs.BOM_UTF8, "utf-8-sig")):
        if data.startswith(marker):
            hint_codecs.append(codec)
            break
    meta = re.search(br"charset\s*=\s*[\"']?\s*([a-zA-Z0-9_-]+)", data[:4096], re.I)
    if meta:
        try:
            hint_codecs.append(codecs.lookup(meta.group(1).decode("ascii")).name)
        except (LookupError, ValueError):
            pass
    candidates.extend(hint_codecs + ["utf-8", "gb18030", "gbk"])
    candidates = list(dict.fromkeys(candidates))
    declared_codec = None
    if declared:
        try:
            declared_codec = codecs.lookup(declared.strip()).name
        except (LookupError, ValueError):
            pass
    for charset in candidates:
        try:
            text = data.decode(charset, errors="strict")
            # A permissive codec can decode binary/control bytes successfully.
            if _control_count(text) / max(1, len(text)) > CONTROL_LIMIT:
                continue
            if charset == declared_codec:
                alternatives = list(hint_codecs)
                if charset in {"iso8859-1", "cp1252"}:
                    alternatives.extend(["utf-8", "gb18030"])
                for other in dict.fromkeys(alternatives):
                    if other == charset:
                        continue
                    try:
                        alternative = data.decode(other, errors="strict")
                    except (UnicodeError, LookupError):
                        continue
                    cjk = _cjk_count(alternative)
                    if (_control_count(alternative) == 0 and _cjk_count(text) == 0
                            and cjk >= 4 and cjk / max(1, len(alternative)) >= 0.15):
                        return Decoded(alternative, declared, other, True, "suspected_misdeclared_charset")
            return Decoded(text, declared, charset, charset != declared_codec)
        except (UnicodeError, LookupError):
            continue
    choices = []
    for charset in candidates:
        try:
            text = data.decode(charset, errors="replace")
            penalty = text.count("\ufffd") + _control_count(text)
            choices.append((penalty, len(choices), text, charset))
        except (UnicodeError, LookupError):
            pass
    if not choices:
        return Decoded(data.decode("utf-8", errors="replace"), declared, "utf-8", True)
    _, _, text, charset = min(choices)
    return Decoded(text, declared, charset, True)


def _payload_bytes(part: Message, actions: list[str]) -> bytes:
    # A shallow copy plus public header deletion preserves raw payload bytes
    # without altering the original message or using its private payload field.
    unencoded = copy.copy(part)
    del unencoded["Content-Transfer-Encoding"]
    raw_value = unencoded.get_payload(decode=True)
    if raw_value is None:
        return b""
    cte = str(part.get("Content-Transfer-Encoding", "")).strip().lower()
    token = cte.split()[0] if cte else ""
    if token in {"base64", "quoted-printable"} and token != cte:
        actions.append("normalized_transfer_encoding")
    if token == "quoted-printable":
        return quopri.decodestring(raw_value)
    if token == "base64":
        # Populate standard-library transfer defects, but do not trust its
        # permissive decoder to turn already-decoded prose into random bytes.
        part.get_payload(decode=True)
        compact = re.sub(rb"\s+", b"", raw_value)
        try:
            return base64.b64decode(compact + b"=" * ((-len(compact)) % 4), validate=True)
        except Exception:
            candidate = _decode_bytes(raw_value, part.get_content_charset())
            text = candidate.text
            visibly_text = (_cjk_count(text) >= 2 or HTML_TAG.search(text) is not None
                            or len(re.findall(r"[A-Za-z]{2,}[ \t]+[A-Za-z]{2,}", text)) >= 2)
            if visibly_text and "\ufffd" not in text and _control_count(text) == 0:
                actions.append("mislabeled_base64_text_fallback")
                return raw_value
            raise ValueError("Invalid base64 without a safe plain-text fallback.")
    if token and token not in {"7bit", "8bit", "binary"}:
        actions.append("unknown_transfer_encoding")
        raise ValueError("Unsupported transfer encoding.")
    return raw_value


def _encoded_blob(text: str) -> bool:
    # Spaces between prose words distinguish it from a wrapped encoded block.
    if re.search(r"[ \t]", text.strip()):
        return False
    compact = re.sub(r"\s+", "", text)
    return len(compact) >= 160 and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", compact) is not None


def _usable_text(text: str) -> bool:
    return any(char.isalnum() for char in text)


def parse_email(raw: bytes) -> dict:
    result = {
        "parser_version": PARSER_VERSION,
        "subject_clean": "", "body_clean": "", "text_clean": "", "body_decoded": "",
        "text_unstripped_html": "", "parse_status": "error", "usable": False,
        "top_mime_type": "", "has_html": False, "body_selection": "none",
        "has_html_part_anywhere": False,
        "mime_defects": [], "recovery_actions": [], "decode_fallback": False,
        "declared_charsets": [], "used_charsets": [], "replacement_ratio": 0.0,
        "replacement_characters": 0, "decoded_characters": 0,
        "retained_characters": 0, "retained_replacement_characters": 0, "retained_replacement_ratio": 0.0,
        "skipped_attachments": 0, "skipped_forwarded": 0,
        "encoding_warnings": [],
        "private_use_characters": 0,
        "subject_missing": True, "body_empty": True, "error_type": "",
        "subject_empty": True,
    }
    actions: list[str] = []
    attempted_decodings: list[Decoded] = []
    rejected_body = False

    def decode(data: bytes, declared: str | None) -> Decoded:
        value = _decode_bytes(data, declared)
        attempted_decodings.append(value)
        return value

    try:
        message = BytesParser(policy=policy.default).parsebytes(raw)
        result["top_mime_type"] = message.get_content_type()
        result["has_html_part_anywhere"] = any(part.get_content_type() == "text/html" for part in message.walk())
        subject = next((value for key, value in message.raw_items() if key.lower() == "subject"), "")
        result["subject_missing"] = not bool(subject.strip())
        subject_parts = []
        try:
            segments = decode_header(re.sub(r"\r?\n[ \t]+", " ", subject))
        except Exception:
            segments = [(subject, None)]
            actions.append("subject_header_recovery")
        for value, charset in segments:
            if isinstance(value, str):
                try:
                    value = value.encode("ascii", errors="surrogateescape")
                except UnicodeEncodeError:
                    subject_parts.append(value)
                    continue
            decoded = decode(value, charset)
            subject_parts.append(decoded.text)
        decoded_subject = "".join(subject_parts)
        if ((decoded_subject.count("\ufffd") > 2
                and decoded_subject.count("\ufffd") / max(1, len(decoded_subject)) > REPLACEMENT_LIMIT)
                or _control_count(decoded_subject) / max(1, len(decoded_subject)) > CONTROL_LIMIT):
            actions.append("subject_candidate_rejected")
        else:
            result["subject_clean"] = clean_text(decoded_subject)
        result["subject_empty"] = not _usable_text(result["subject_clean"])

        def extract(part: Message, depth: int = 0) -> tuple[str, str, str, bool]:
            nonlocal rejected_body
            if depth > 20:
                actions.append("mime_depth_limit")
                return "", "", "none", False
            if part.get_content_maintype() == "message":
                result["skipped_forwarded"] += 1
                if part.get_content_disposition() == "attachment" or part.get_filename() is not None:
                    result["skipped_attachments"] += 1
                return "", "", "none", False
            if part.get_content_disposition() == "attachment" or part.get_filename() is not None:
                result["skipped_attachments"] += 1
                return "", "", "none", False
            if part.is_multipart():
                children = part.get_payload()
                if part.get_content_subtype() == "related":
                    start = part.get_param("start", header="content-type")
                    root = children[0]
                    if start:
                        matching = [child for child in children if str(child.get("Content-ID", "")).strip() == str(start).strip()]
                        if matching:
                            root = matching[0]
                        else:
                            actions.append("related_start_not_found")
                    return extract(root, depth + 1)
                if part.get_content_subtype() == "alternative":
                    # Examine plain parts first; only decode fallback HTML if needed.
                    ordered = sorted(children, key=lambda child: child.get_content_type() != "text/plain")
                    for child in ordered:
                        decoded_text, cleaned, selection, has_html = extract(child, depth + 1)
                        if _usable_text(cleaned):
                            return decoded_text, cleaned, selection, has_html
                    return "", "", "none", False
                selected = [extract(child, depth + 1) for child in children]
                selected = [item for item in selected if _usable_text(item[1])]
                unique = []
                seen = set()
                for item in selected:
                    if item[1] not in seen:
                        unique.append(item)
                        seen.add(item[1])
                    else:
                        actions.append("duplicate_body_representation")
                selected = unique
                return (
                    "\n".join(item[0] for item in selected),
                    "\n".join(item[1] for item in selected),
                    "mixed" if len(selected) > 1 else (selected[0][2] if selected else "none"),
                    any(item[3] for item in selected),
                )
            broken_multipart = part.get_content_maintype() == "multipart"
            if not broken_multipart and part.get_content_type() not in {"text/plain", "text/html"}:
                result["skipped_attachments"] += 1
                return "", "", "none", False
            try:
                data = _payload_bytes(part, actions)
            except Exception:
                actions.append("transfer_decode_failed")
                rejected_body = True
                return "", "", "rejected", False
            decoded = decode(data, part.get_content_charset())
            if broken_multipart:
                actions.append("broken_multipart_text_fallback")
            text = decoded.text
            replacement_ratio = text.count("\ufffd") / max(1, len(text))
            binary_like = _control_count(text) / max(1, len(text)) > CONTROL_LIMIT
            boundary = part.get_boundary()
            exact_boundary = bool(boundary and re.search(
                r"(?m)^--" + re.escape(boundary) + r"(?:--)?[ \t]*$", text.replace("\r\n", "\n")
            ))
            framed = broken_multipart and (exact_boundary or MIME_HEADER_BLOCK.search(text) is not None)
            if replacement_ratio > REPLACEMENT_LIMIT or binary_like or _encoded_blob(text) or framed:
                actions.append("body_candidate_rejected")
                rejected_body = True
                return text, "", "rejected", False
            is_html = part.get_content_type() == "text/html" or HTML_TAG.search(text) is not None
            readable = html_to_text(text) if is_html else text
            return text, clean_text(readable), "html" if is_html else "plain", is_html

        body_decoded, body_clean, selection, has_html = extract(message)
        result["body_decoded"] = body_decoded
        result["body_clean"] = body_clean
        result["body_selection"] = selection
        result["has_html"] = has_html and _usable_text(body_clean)
        result["body_empty"] = not _usable_text(body_clean)
        result["text_clean"] = "\n".join(value for value in (result["subject_clean"], body_clean) if value).strip()
        result["text_unstripped_html"] = "\n".join(
            value for value in (result["subject_clean"], clean_text(body_decoded) if body_clean else "") if value
        ).strip()
        result["mime_defects"] = sorted({type(defect).__name__ for part in message.walk() for defect in part.defects})
        result["recovery_actions"] = sorted(set(actions))
        result["decode_fallback"] = any(item.fallback for item in attempted_decodings)
        result["declared_charsets"] = sorted({item.declared for item in attempted_decodings if item.declared})
        result["used_charsets"] = sorted({item.used for item in attempted_decodings})
        result["encoding_warnings"] = sorted({item.warning for item in attempted_decodings if item.warning})
        decoded_text = "".join(item.text for item in attempted_decodings)
        result["private_use_characters"] = sum(unicodedata.category(char) == "Co" for char in decoded_text)
        if result["private_use_characters"]:
            result["encoding_warnings"] = sorted(set(result["encoding_warnings"]) | {"private_use_characters_present"})
        result["decoded_characters"] = len(decoded_text)
        result["replacement_characters"] = decoded_text.count("\ufffd")
        result["replacement_ratio"] = result["replacement_characters"] / max(1, len(decoded_text))
        result["retained_characters"] = len(result["text_clean"])
        result["retained_replacement_characters"] = result["text_clean"].count("\ufffd")
        result["retained_replacement_ratio"] = result["retained_replacement_characters"] / max(1, result["retained_characters"])
        result["usable"] = _usable_text(result["text_clean"])
        if not result["usable"]:
            result["parse_status"] = "quarantined" if rejected_body or "subject_candidate_rejected" in actions else "empty"
        elif result["body_empty"]:
            result["parse_status"] = "subject_only"
        elif result["mime_defects"] or actions or result["decode_fallback"] or result["replacement_characters"] or result["encoding_warnings"]:
            result["parse_status"] = "recovered"
        else:
            result["parse_status"] = "ok"
        return result
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["parse_status"] = "error"
        result["usable"] = False
        result["text_clean"] = ""
        result["body_clean"] = ""
        result["recovery_actions"] = sorted(set(actions))
        return result
