"""Reranker default logic. Run: python -m pytest tests/test_retriever_config.py -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import should_use_reranker


def test_absent_key_keeps_old_behaviour():
    assert should_use_reranker(None, {}) is True


def test_config_false_turns_default_off():
    assert should_use_reranker(None, {"use_reranker": False}) is False


def test_config_true():
    assert should_use_reranker(None, {"use_reranker": True}) is True


def test_explicit_request_overrides_config_both_ways():
    assert should_use_reranker(True, {"use_reranker": False}) is True
    assert should_use_reranker(False, {"use_reranker": True}) is False
    assert should_use_reranker(False, {}) is False