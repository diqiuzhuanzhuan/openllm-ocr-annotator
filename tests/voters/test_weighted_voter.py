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

import pytest
from src.openllm_ocr_annotator.voters.weighted import WeightedVoter


def _make_annotation(fields):
    """Helper to build an annotation dict in the expected structure."""
    return {
        "result": {
            "fields": [
                {"field_name": k, "value": v, "confidence": 1.0}
                for k, v in fields.items()
            ]
        }
    }


class TestWeightedVoterGetWeight:
    def test_known_annotator(self):
        voter = WeightedVoter(weights={"annotator_a/model_x": 2.0})
        assert voter.get_weight("annotator_a/model_x") == 2.0

    def test_unknown_annotator_uses_default(self):
        voter = WeightedVoter(weights={})
        assert voter.get_weight("unknown/model") == 1.0

    def test_sampling_suffix_stripped(self):
        voter = WeightedVoter(weights={"annotator_a/model_x": 2.0})
        weight = voter.get_weight("annotator_a/model_x/sample_3", num_samples=3)
        assert weight == 2.0


class TestWeightedVoterVote:
    def test_empty_annotations_raises(self):
        voter = WeightedVoter()
        with pytest.raises(ValueError):
            voter.vote([], [])

    def test_single_annotator_full_confidence(self):
        voter = WeightedVoter()
        annotation = _make_annotation({"invoice_number": "INV-001"})
        result = voter.vote([annotation], ["annotator_a/model_x"])
        fields = {f["field_name"]: f for f in result["fields"]}
        assert fields["invoice_number"]["value"] == "INV-001"
        assert fields["invoice_number"]["confidence"] == pytest.approx(1.0)

    def test_two_annotators_agree(self):
        voter = WeightedVoter()
        ann = _make_annotation({"total": "100.00"})
        result = voter.vote([ann, ann], ["a/m1", "a/m2"])
        fields = {f["field_name"]: f for f in result["fields"]}
        assert fields["total"]["value"] == "100.00"
        assert fields["total"]["confidence"] == pytest.approx(1.0)

    def test_two_annotators_disagree_higher_weight_wins(self):
        weights = {"high/model": 2.0, "low/model": 0.5}
        voter = WeightedVoter(weights=weights)
        ann_high = _make_annotation({"field": "value_A"})
        ann_low = _make_annotation({"field": "value_B"})
        result = voter.vote([ann_high, ann_low], ["high/model", "low/model"])
        fields = {f["field_name"]: f for f in result["fields"]}
        assert fields["field"]["value"] == "value_A"

    def test_mismatched_lengths_raises(self):
        voter = WeightedVoter()
        ann = _make_annotation({"k": "v"})
        with pytest.raises(ValueError):
            voter.vote([ann], ["id_a", "id_b"])

    def test_confidence_is_normalized(self):
        voter = WeightedVoter(weights={"a/m": 3.0, "b/m": 1.0})
        ann_a = _make_annotation({"field": "X"})
        ann_b = _make_annotation({"field": "Y"})
        result = voter.vote([ann_a, ann_b], ["a/m", "b/m"])
        fields = {f["field_name"]: f for f in result["fields"]}
        # winner X has weight 3.0 out of total 4.0
        assert fields["field"]["confidence"] == pytest.approx(3.0 / 4.0)
