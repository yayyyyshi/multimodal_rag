"""Deterministic answer formatting.

This module only cleans presentation: whitespace, wrapped list lines,
repeated lines or sentences, and a final sentence cut off by the token
limit. It does NOT check whether an answer is supported by the evidence
and must never be described as verification.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

CITE_RE = re.compile(r"\[(\d+)\]")

_NUM_RE = re.compile(r"^\s*\d+[.)]\s+")
_BUL_RE = re.compile(r"^\s*[-*\u2022]\s+")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# A sentence end: terminator, optional closing quote/bracket, optional
# trailing [n] markers, followed by whitespace or end of text.
_SENT_END_RE = re.compile(
    r"[.!?][\"')\]\u201d\u2019]*(?:\s*\[\d+\])*(?=\s|$)"
)

DEFAULT_ABSTAIN = "Not answerable"
_ABSTAIN_ALIASES = {"", "unknown", "i don't know", "i do not know"}

JACCARD_DUP = 0.55
SEQUENCE_DUP = 0.78
MIN_TOKENS_FOR_SENTENCE_DEDUP = 4


def _normalise(text: str) -> str:
    text = CITE_RE.sub("", text)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _is_near_duplicate(candidate: str, previous: list[str]) -> bool:
    """Exact or near-duplicate list items (same thresholds as before)."""
    if not candidate:
        return False
    tokens = set(candidate.split())

    for old in previous:
        if candidate == old:
            return True
        old_tokens = set(old.split())
        if not tokens or not old_tokens:
            continue
        jaccard = len(tokens & old_tokens) / len(tokens | old_tokens)
        ratio = SequenceMatcher(None, candidate, old).ratio()
        if jaccard >= JACCARD_DUP or ratio >= SEQUENCE_DUP:
            return True
    return False


def _parse_units(text: str) -> list[dict]:
    """Split text into list items and paragraphs, keeping order.

    Wrapped continuation lines stay attached to their list item; blank
    lines are layout only and are discarded here and rebuilt on render.
    """
    units: list[dict] = []
    prev_blank = False

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()

        if not line.strip():
            prev_blank = True
            continue

        match_num = _NUM_RE.match(line)
        match_bul = _BUL_RE.match(line)

        if match_num or match_bul:
            match = match_num or match_bul
            units.append({
                "kind": "item",
                "marker": "num" if match_num else "bullet",
                "text": line[match.end():].strip(),
            })
        else:
            last = units[-1] if units else None
            indented = raw_line[:1] in (" ", "\t")

            if last and last["kind"] == "item" and (not prev_blank or indented):
                last["text"] += " " + line.strip()
            elif last and last["kind"] == "para" and not prev_blank:
                last["text"] += " " + line.strip()
            else:
                units.append({
                    "kind": "para",
                    "marker": None,
                    "text": line.strip(),
                })

        prev_blank = False

    return units


def _drop_repeats(units: list[dict], stats: dict) -> list[dict]:
    """Drop exact repeated sentences and near-duplicate list items."""
    kept: list[dict] = []
    item_norms: list[str] = []
    seen_sentences: set[str] = set()

    for unit in units:
        fresh = []
        for sentence in _SENT_SPLIT_RE.split(unit["text"]):
            norm = _normalise(sentence)
            long_enough = len(norm.split()) >= MIN_TOKENS_FOR_SENTENCE_DEDUP

            if long_enough and norm in seen_sentences:
                stats["removed_duplicates"] += 1
                continue
            if long_enough:
                seen_sentences.add(norm)
            fresh.append(sentence)

        if not fresh:
            continue

        unit["text"] = " ".join(fresh)

        if unit["kind"] == "item":
            norm = _normalise(unit["text"])
            if _is_near_duplicate(norm, item_norms):
                stats["removed_duplicates"] += 1
                continue
            item_norms.append(norm)

        kept.append(unit)

    return kept


def _trim_incomplete_tail(units: list[dict], stats: dict) -> list[dict]:
    """Cut a final sentence that the token limit left unfinished."""
    if not units:
        return units

    text = units[-1]["text"].rstrip()
    ends = list(_SENT_END_RE.finditer(text))

    if ends and ends[-1].end() == len(text):
        return units  # already ends cleanly

    if ends:
        units[-1]["text"] = text[: ends[-1].end()]
    else:
        units = units[:-1]

    stats["trimmed_incomplete"] = True
    return units


def _render(units: list[dict]) -> str:
    parts: list[str] = []
    prev = None
    number = 0

    for unit in units:
        if unit["kind"] == "item":
            if unit["marker"] == "num":
                number = number + 1 if prev == "num" else 1
                line = f"{number}. {unit['text']}"
                kind = "num"
            else:
                line = f"- {unit['text']}"
                kind = "bullet"
            sep = "\n" if prev in ("num", "bullet") else "\n\n"
        else:
            line, kind, sep = unit["text"], "para", "\n\n"

        parts.append(line if prev is None else sep + line)
        prev = kind

    return "".join(parts).strip()


def clean_answer(answer: str, truncated: bool = False) -> tuple[str, dict]:
    """Return (cleaned_text, stats). Formatting only, never fact-checking.

    stats = {"removed_duplicates": int, "trimmed_incomplete": bool}
    There is no cap on the number of list items.
    """
    stats = {"removed_duplicates": 0, "trimmed_incomplete": False}
    original = (answer or "").strip()

    if not original:
        return original, stats

    text = _BOLD_RE.sub(
        lambda m: m.group(1) or m.group(2),
        original.replace("\r\n", "\n"),
    )
    units = _parse_units(text)
    if not units:
        return original, stats

    units = _drop_repeats(units, stats)
    if truncated:
        units = _trim_incomplete_tail(units, stats)

    if not units:
        return original, stats

    return _render(units), stats


def normalise_abstention(text: str, abstain: str = DEFAULT_ABSTAIN) -> str:
    """Return exactly `abstain` when the whole answer is just an abstention.

    "Not answerable [6]" or "Not answerable." become "Not answerable", so a
    stray [n] marker cannot turn an abstention into an apparent source. Text
    that merely *starts* with the abstention phrase and continues with other
    content is returned unchanged.
    """
    bare = CITE_RE.sub("", text or "").strip(" \t\r\n.!")
    if bare.lower() == abstain.lower() or bare.lower() in _ABSTAIN_ALIASES:
        return abstain
    return text