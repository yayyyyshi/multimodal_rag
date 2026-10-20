"""Formatting regression tests. Run: python -m unittest tests.test_postprocess -v

These test presentation cleanup only. They do not (and cannot) test
whether an answer is faithful to the retrieved evidence.
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.postprocess import clean_answer, normalise_abstention

# Real raw outputs captured from scripts/diag_raw.py (Germanwings query).
RAW_RUN1 = (
    'During the first hours of the crisis, Germanwings communicated updates through the following methods:\n\n'
    '1. **Twitter**: Germanwings issued its first tweet acknowledging the incident and providing an update to '
    'Facebook within the first 60 minutes. This was done within the first 60 minutes following the incident, '
    'as indicated by the timestamp "25-03-2015 a y BRENDAN HODGSON + AHMED ALSHARIOI" in the context.\n\n'
    '2. **Social Platforms**: Germanwings\' website and social platforms, including Twitter and Facebook, were '
    'inaccessible for approximately 2 hours. This period was from the time of the incident until the '
    'availability of the first update on the website, as mentioned in the context.\n\n'
    '3. **Website**: The Germanwings website remained inaccessible for approximately 2 hours, as indicated by '
    'the context.\n\n'
    '4. **Updates**: The website and social platforms were updated periodically, as mentioned in the context. '
    'The updates were available on the website and through Twitter and Facebook, as indicated by the context.\n\n'
    '5. **Twitter**: Germanwings continued to monitor its website for periodic updates, as indicated by the '
    'context.\n\n'
    'These methods were used to communicate updates during the first hours of the crisis, highlighting the importance'
)

RAW_RUN2 = (
    "During the first hours of the crisis, Germanwings communicated updates through the following methods:\n\n"
    "1. Germanwings' website remained inaccessible for approximately 2 hours, making it only accessible for a short period.\n"
    "2. Germanwings' social media platforms, including Twitter and Facebook, became the primary points of digital contact with the public. Twitter and Facebook became only available within the first hour of the crisis.\n"
    "3. Germanwings' website was inaccessible for approximately 2 hours, but Twitter and Facebook remained available as points of contact with the public.\n"
    "4. Germanwings' website was inaccessible for approximately 2 hours, but Twitter and Facebook remained available as points of contact with the public.\n"
    "5. Germanwings' website remained inaccessible for approximately 2 hours, but Twitter and Facebook remained available as points of contact with the public.\n"
    "6. Germanwings' website remained inaccessible for approximately 2 hours, but Twitter and Facebook remained available as points of contact with the public.\n"
    "7. Germanwings' website remained inaccessible for approximately 2 hours, but Twitter and Facebook remained available as points of contact with the public.\n"
    "8. Germanwings' website remained inaccessible for approximately 2 hours, but Twitter and Facebook remained available as points of contact with the public."
)


def numbered_lines(text):
    return [l for l in text.splitlines() if re.match(r"^\d+\. ", l)]


class StructureTests(unittest.TestCase):
    def assert_well_formed(self, text):
        self.assertEqual(text, text.strip(), "leading/trailing whitespace")
        self.assertNotIn("\n\n\n", text, "more than one consecutive blank line")
        for line in text.splitlines():
            self.assertEqual(line, line.lstrip(), f"indented line: {line!r}")
        nums = [int(re.match(r"^(\d+)\.", l).group(1)) for l in numbered_lines(text)]
        self.assertEqual(nums, list(range(1, len(nums) + 1)), "numbering not 1..n")

    def test_real_run1_truncated(self):
        out, stats = clean_answer(RAW_RUN1, truncated=True)
        self.assert_well_formed(out)
        self.assertTrue(out.startswith("During the first hours"))
        self.assertNotIn("**", out)
        # The cut-off closing sentence must be removed, not relocated.
        self.assertNotIn("highlighting the importance", out)
        self.assertNotIn("These methods were used", out)
        self.assertTrue(stats["trimmed_incomplete"])
        # Intro stays above the list; list items are in original order.
        first_item = out.index("1. ")
        self.assertLess(out.index("During the first hours"), first_item)

    def test_real_run1_not_marked_truncated_keeps_tail(self):
        out, stats = clean_answer(RAW_RUN1, truncated=False)
        self.assertFalse(stats["trimmed_incomplete"])
        self.assertTrue(out.endswith("highlighting the importance"))
        self.assert_well_formed(out)

    def test_real_run2_repetition_collapsed(self):
        out, stats = clean_answer(RAW_RUN2, truncated=False)
        self.assert_well_formed(out)
        self.assertLessEqual(len(numbered_lines(out)), 3)
        self.assertGreaterEqual(stats["removed_duplicates"], 4)

    def test_blank_lines_and_wrapped_continuation(self):
        raw = (
            "Germanwings communicated in several ways:\n\n"
            "1. Within the first 60 minutes, Germanwings issued its first tweet [2]\n"
            "   and updated Facebook.\n\n"
            "2. The website crashed and remained inaccessible for about two hours [2].\n\n"
            "3. By roughly hour five a crisis website was in place [1].\n"
        )
        out, _ = clean_answer(raw)
        self.assert_well_formed(out)
        self.assertIn("first tweet [2] and updated Facebook.", out)
        self.assertEqual(len(numbered_lines(out)), 3)

    def test_no_silent_cap_on_distinct_items(self):
        raw = "\n".join([
            "1. The first tweet was issued within sixty minutes [2].",
            "2. The crisis website was live around hour five [1].",
            "3. Facebook was updated during the opening hour [2].",
            "4. The corporate homepage crashed under traffic [2].",
            "5. Customers were directed to the crisis page for periodic updates [1].",
            "6. Lufthansa later published a joint statement with its subsidiary [4].",
        ])
        out, stats = clean_answer(raw)
        self.assertEqual(len(numbered_lines(out)), 6)
        self.assertEqual(stats["removed_duplicates"], 0)

    def test_distinct_overlapping_facts_not_merged(self):
        raw = (
            "1. The Germanwings website crashed and remained inaccessible for approximately two hours [2].\n"
            "2. During the first hour, Twitter and Facebook were the only available digital contact points [6]."
        )
        out, stats = clean_answer(raw)
        self.assertEqual(len(numbered_lines(out)), 2)
        self.assertEqual(stats["removed_duplicates"], 0)

    def test_prose_repeated_sentence_removed(self):
        raw = (
            "The website was inaccessible for about two hours [2]. "
            "Twitter and Facebook stayed available [6]. "
            "The website was inaccessible for about two hours [2]."
        )
        out, stats = clean_answer(raw)
        self.assertEqual(out.count("inaccessible for about two hours"), 1)
        self.assertEqual(stats["removed_duplicates"], 1)

    def test_plain_short_answer_untouched(self):
        raw = "The website was down for about two hours [2]."
        out, stats = clean_answer(raw, truncated=True)
        self.assertEqual(out, raw)
        self.assertFalse(stats["trimmed_incomplete"])

    def test_decimal_and_abbrev_not_treated_as_sentence_end(self):
        raw = "The first tweet came within 2.5 hours and the page was updated at p.12 and then"
        out, _ = clean_answer(raw, truncated=True)
        # no sentence terminator followed by whitespace -> whole unit dropped
        # only if nothing salvageable; here nothing salvageable, so original kept
        self.assertEqual(out, raw)

    def test_idempotent(self):
        for raw, trunc in ((RAW_RUN1, True), (RAW_RUN2, False)):
            once, _ = clean_answer(raw, truncated=trunc)
            twice, _ = clean_answer(once, truncated=trunc)
            self.assertEqual(once, twice)

    def test_empty_input(self):
        self.assertEqual(clean_answer("")[0], "")
        self.assertEqual(clean_answer("   \n\n ")[0], "")


class AbstentionTests(unittest.TestCase):
    def test_marker_after_abstention_removed(self):
        # Real 3B output: "Not answerable [6]" previously created a citation.
        self.assertEqual(normalise_abstention("Not answerable [6]"), "Not answerable")

    def test_variants_normalised(self):
        for text in ("Not answerable", "not answerable.", "  Not answerable [1][2] ", "unknown", "", "I don't know."):
            self.assertEqual(normalise_abstention(text), "Not answerable", text)

    def test_abstention_followed_by_content_is_left_alone(self):
        text = "Not answerable. However, the website was down for about two hours [2]."
        self.assertEqual(normalise_abstention(text), text)

    def test_real_answers_untouched(self):
        text = "The website was down for about two hours [2]."
        self.assertEqual(normalise_abstention(text), text)

    def test_custom_phrase(self):
        self.assertEqual(normalise_abstention("No answer [3]", "No answer"), "No answer")


if __name__ == "__main__":
    unittest.main()