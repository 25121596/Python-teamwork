"""Fixed jieba tokenization and a training-fitted sparse TF-IDF pipeline."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STOPWORDS_PATH = PROJECT_ROOT / "config/stopwords_zh.txt"
PUBLIC_TERMS_PATH = PROJECT_ROOT / "config/public_terms_zh.txt"


def read_terms(path: Path) -> frozenset[str]:
    return frozenset(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                     if line.strip() and not line.lstrip().startswith("#"))


STOPWORDS = read_terms(STOPWORDS_PATH)
PUBLIC_TERMS = read_terms(PUBLIC_TERMS_PATH)
PLACEHOLDERS = frozenset({"urltoken", "emailtoken", "phonetoken"})
TOKEN_PATTERN = re.compile(r"[a-z0-9\u3400-\u9fff]+\Z")
TOKENIZER = jieba.Tokenizer()
jieba.setLogLevel(logging.ERROR)


def tokenize_text(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens = []
    for value in TOKENIZER.cut(text, cut_all=False, HMM=False):
        token = value.strip()
        if token in PLACEHOLDERS:
            tokens.append(token)
        elif (token not in STOPWORDS and TOKEN_PATTERN.fullmatch(token)
              and len(token) <= 40 and not (len(token) == 1 and token.isascii() and token.isalpha())):
            tokens.append(token)
    return tokens


def make_feature_pipeline() -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(
        tokenizer=tokenize_text, token_pattern=None, lowercase=False,
        ngram_range=(1, 1), min_df=2, max_df=1.0, max_features=50000,
        sublinear_tf=True, use_idf=True, smooth_idf=True, norm="l2", dtype=np.float64,
    ))])


def tokenizer_fingerprint() -> dict:
    # The bundled default dictionary is fixed with the pinned jieba package;
    # do not expose the local package path in public metadata.
    with TOKENIZER.get_dict_file() as dictionary:
        dictionary_sha = hashlib.sha256(dictionary.read()).hexdigest()
    return {
        "jieba_version": jieba.__version__, "dictionary_sha256": dictionary_sha,
        "stopwords_sha256": hashlib.sha256(STOPWORDS_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "public_terms_sha256": hashlib.sha256(PUBLIC_TERMS_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "stopword_count": len(STOPWORDS), "public_term_count": len(PUBLIC_TERMS),
        "mode": "accurate", "HMM": False, "max_token_length": 40,
    }
