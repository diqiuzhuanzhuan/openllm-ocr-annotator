# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT


from openllm_ocr_annotator.voters.base import BaseVoter
from typing import List, Dict
from collections import Counter


class MajorityVoter(BaseVoter):
    def vote(self, annotations: List[Dict]) -> Dict:
        """Implement majority voting strategy."""
        results = {}
        for field in annotations[0].keys():
            values = [ann[field] for ann in annotations]
            counter = Counter(values)
            results[field] = counter.most_common(1)[0][0]
        return results
