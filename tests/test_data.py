"""Tests for data utilities that don't require network access.

Testing ``load_vqarad`` itself requires hitting the HuggingFace hub, so we
exercise the pure-Python helpers here. The integration tests in the
notebooks cover the network path.
"""

from __future__ import annotations

import pytest

from src.data.load_vqarad import classify_question_type


class TestClassifyQuestionType:
    @pytest.mark.parametrize("answer", [
        "yes", "no", "Yes", "NO", "YES", "yes.", "no!",
        "  yes  ", "Yes,", "no, definitely not".split(",")[0].strip(),
    ])
    def test_closed_answers(self, answer):
        assert classify_question_type(answer) == "closed"

    @pytest.mark.parametrize("answer", [
        "cardiomegaly", "left lung", "the patient has pneumonia",
        "yes and no",  # multi-token, not closed
        "perhaps", "uncertain",
    ])
    def test_open_answers(self, answer):
        assert classify_question_type(answer) == "open"

    def test_handles_none(self):
        assert classify_question_type(None) == "open"

    def test_handles_empty(self):
        assert classify_question_type("") == "open"

    def test_handles_whitespace_only(self):
        assert classify_question_type("   ") == "open"
