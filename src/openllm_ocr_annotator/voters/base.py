# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseVoter(ABC):
    @abstractmethod
    def vote(self, annotations: List[Dict]) -> Dict:
        """Base voting method interface."""
        pass
