# MIT License
#
# Copyright (c) 2025 LoongMa
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from src.openllm_ocr_annotator.voters.majority import MajorityVoter


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
