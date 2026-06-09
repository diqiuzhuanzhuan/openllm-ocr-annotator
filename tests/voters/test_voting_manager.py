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


import json
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from openllm_ocr_annotator.voters.manager import VotingManager


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class TestVotingManagerCollectAnnotations:
    def setup_method(self):
        self.voter = MagicMock()

    def test_single_mode_loads_result(self, tmp_path):
        # Write a fixture result file
        ann_dir = tmp_path / "annotator_a" / "gpt-4"
        ann_dir.mkdir(parents=True)
        result = {"result": {"fields": []}, "model": "gpt-4"}
        _write_json(ann_dir / "image001.json", result)

        manager = VotingManager(
            annotator_infos=[
                {"name": "annotator_a", "model": "gpt-4", "results_dir": tmp_path}
            ],
            voter=self.voter,
        )
        results = manager.collect_annotations(
            results_dir=tmp_path, image_stem="image001", num_samples=1
        )
        assert "annotator_a/gpt-4" in results
        assert results["annotator_a/gpt-4"] == result

    def test_missing_file_skipped(self, tmp_path):
        manager = VotingManager(
            annotator_infos=[
                {"name": "annotator_a", "model": "gpt-4", "results_dir": tmp_path}
            ],
            voter=self.voter,
        )
        results = manager.collect_annotations(
            results_dir=tmp_path, image_stem="nonexistent", num_samples=1
        )
        assert results == {}

    def test_sampling_mode_loads_multiple_samples(self, tmp_path):
        for i in range(3):
            sample_dir = tmp_path / "annotator_a" / "gpt-4" / "sampling" / f"sample_{i}"
            sample_dir.mkdir(parents=True)
            _write_json(sample_dir / "image001.json", {"result": {}, "sample": i})

        manager = VotingManager(
            annotator_infos=[
                {"name": "annotator_a", "model": "gpt-4", "results_dir": tmp_path}
            ],
            voter=self.voter,
        )
        results = manager.collect_annotations(
            results_dir=tmp_path, image_stem="image001", num_samples=3
        )
        assert len(results) == 3
        assert "annotator_a/gpt-4/sample_0" in results
        assert "annotator_a/gpt-4/sample_2" in results


class TestVotingManagerGetVotedResult:
    def test_calls_voter_and_returns_result(self, tmp_path):
        # Write fixture result file
        ann_dir = tmp_path / "annotator_a" / "gpt-4"
        ann_dir.mkdir(parents=True)
        annotation = {
            "result": {"fields": [{"field_name": "k", "value": "v", "confidence": 1.0}]}
        }
        _write_json(ann_dir / "image001.json", annotation)

        mock_voter = MagicMock()
        mock_voter.vote.return_value = {
            "fields": [{"field_name": "k", "value": "v", "confidence": 1.0}]
        }

        manager = VotingManager(
            annotator_infos=[
                {"name": "annotator_a", "model": "gpt-4", "results_dir": tmp_path}
            ],
            voter=mock_voter,
        )
        img_path = Path("image001.jpg")
        result = manager.get_voted_result(img_path, output_dir=tmp_path, num_samples=1)

        assert "result" in result
        mock_voter.vote.assert_called_once()

    def test_no_annotations_raises(self, tmp_path):
        mock_voter = MagicMock()
        manager = VotingManager(
            annotator_infos=[
                {"name": "annotator_a", "model": "gpt-4", "results_dir": tmp_path}
            ],
            voter=mock_voter,
        )
        with pytest.raises(ValueError, match="No valid annotations"):
            manager.get_voted_result(
                Path("missing.jpg"), output_dir=tmp_path, num_samples=1
            )
