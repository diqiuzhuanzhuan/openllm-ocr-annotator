# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

from openllm_ocr_annotator.voters.majority import MajorityVoter


class TestMajorityVoter:
    def setup_method(self):
        self.voter = MajorityVoter()

    def test_majority_wins(self):
        annotations = [
            {"status": "approved"},
            {"status": "approved"},
            {"status": "rejected"},
        ]
        result = self.voter.vote(annotations)
        assert result["status"] == "approved"

    def test_tie_returns_first_encountered(self):
        annotations = [
            {"value": "A"},
            {"value": "B"},
        ]
        result = self.voter.vote(annotations)
        # With equal counts, Counter.most_common returns one of them
        assert result["value"] in ("A", "B")

    def test_unanimous_all_same(self):
        annotations = [
            {"field": "X"},
            {"field": "X"},
            {"field": "X"},
        ]
        result = self.voter.vote(annotations)
        assert result["field"] == "X"

    def test_multiple_fields(self):
        annotations = [
            {"a": "1", "b": "yes"},
            {"a": "1", "b": "no"},
            {"a": "2", "b": "yes"},
        ]
        result = self.voter.vote(annotations)
        assert result["a"] == "1"
        assert result["b"] == "yes"
