"""CJK-aware text normalization helpers.

`\b` word boundaries do not work around CJK characters, so matchers that rely
on them silently fail for Chinese text. This module provides:

- header normalization (strip punctuation / whitespace / bullets)
- CJK detection
- term-pattern construction that uses `\b` for ASCII and exact/lookaround
  matching for CJK
- tokenization for overlap classification that includes ASCII words and CJK
  bigrams
"""

from __future__ import annotations

import re
import unicodedata

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Characters commonly surrounding section headers that should be stripped.
_HEADER_NOISE_RE = re.compile(
    r"^[\s\u3000:：\-—–－~～_·•▪‣\[\]【】]+|[\s\u3000:：\-—–－~～_·•▪‣\[\]【】]+$"
)

# ASCII tokenization for overlap classification.
_ASCII_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[+./]?[a-zA-Z0-9]+)*")


def contains_cjk(text: str) -> bool:
    """Return True if text contains a CJK unified ideograph."""
    return _CJK_RE.search(text) is not None


def normalize_header(line: str) -> str:
    """Normalize a line so section headers can be matched robustly.

    Steps:
    - Unicode NFKC normalization
    - strip leading/trailing whitespace and common header punctuation/bullets
    - lowercase for ASCII portions
    """
    line = unicodedata.normalize("NFKC", line)
    line = _HEADER_NOISE_RE.sub("", line)
    # Lowercase only ASCII characters so CJK headers stay intact.
    return line.lower()


def build_term_pattern(term: str) -> str:
    """Return a regex pattern that matches `term` with appropriate boundaries.

    For ASCII terms, use `\b` word boundaries. For terms containing CJK,
    use exact substring matching — CJK does not have useful word boundaries,
    and the caller is responsible for ordering longer terms first when needed.
    """
    if contains_cjk(term):
        return re.escape(term)
    return rf"\b{re.escape(term)}\b"


def token_set(text: str) -> set[str]:
    """Return a set of overlap tokens for text classification.

    Includes ASCII words/numbers and overlapping CJK bigrams. Bigrams capture
    local semantic signal for short Chinese phrases without requiring a
    dictionary.
    """
    text = unicodedata.normalize("NFKC", text)
    tokens: set[str] = set()

    for match in _ASCII_TOKEN_RE.finditer(text):
        tokens.add(match.group(0).lower())

    # Extract CJK characters and form overlapping bigrams.
    cjk_chars = _CJK_RE.findall(text)
    for i in range(len(cjk_chars) - 1):
        tokens.add(cjk_chars[i] + cjk_chars[i + 1])

    return tokens
