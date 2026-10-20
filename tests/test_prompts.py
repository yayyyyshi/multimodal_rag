"""Prompt contract tests. Run: python -m unittest tests.test_prompts -v"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.generator import ExtractiveGenerator
from src.generation.prompts import (
    ABSTAIN,
    SYSTEM_RAG,
    build_no_rag_messages,
    build_rag_messages,
    format_context,
)

ITEMS = [
    dict(doc_id="d", page_num=12, modality="text", content="Alpha beta gamma."),
    dict(doc_id="d", page_num=6, modality="table", content="Row one | Row two"),
    dict(doc_id="d", page_num=9, modality="image", caption="A bar chart."),
]


class PromptContractTests(unittest.TestCase):
    def test_roles_and_labels(self):
        m = build_rag_messages("Q?", ITEMS)
        self.assertEqual([x["role"] for x in m], ["system", "user"])
        self.assertIn("Context:", m[1]["content"])
        self.assertIn("\n\nQuestion: Q?", m[1]["content"])
        self.assertTrue(m[1]["content"].rstrip().endswith("Answer:"))

    def test_numbering_matches_item_order(self):
        text = format_context(ITEMS)
        self.assertLess(text.index("[1] (d, page 12, text passage)"),
                        text.index("[2] (d, page 6, table)"))
        self.assertIn("[3] (d, page 9, figure description)", text)

    def test_abstain_phrase_in_system_prompt(self):
        self.assertIn(ABSTAIN, SYSTEM_RAG)

    def test_system_prompt_stays_short(self):
        # Guards against prompt bloat: ~2,500 chars before Patch B1.
        self.assertLess(len(SYSTEM_RAG), 1500)

    def test_citation_rule_stated_once(self):
        self.assertEqual(SYSTEM_RAG.count("like [2]"), 1)

    def test_weak_warning_only_when_weak(self):
        self.assertNotIn("weakly related", build_rag_messages("Q?", ITEMS)[1]["content"])
        self.assertIn("weakly related", build_rag_messages("Q?", ITEMS, weak=True)[1]["content"])

    def test_extractive_backend_compatible(self):
        gen = ExtractiveGenerator()
        self.assertEqual(gen.generate(build_rag_messages("Q?", ITEMS)), "Alpha beta gamma. [1]")
        self.assertEqual(gen.generate(build_rag_messages("Q?", ITEMS, weak=True)), ABSTAIN)

    def test_no_rag_unchanged(self):
        m = build_no_rag_messages("Q?")
        self.assertEqual(m[1]["content"], "Q?")
        self.assertIn("general knowledge", m[0]["content"])

    def test_empty_content_placeholder(self):
        text = format_context([dict(doc_id="d", page_num=1, modality="text", content="")])
        self.assertIn("No text content available", text)


if __name__ == "__main__":
    unittest.main()