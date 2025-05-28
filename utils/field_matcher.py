# Copyright (c) 2025 Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from abc import ABC, abstractmethod
import re
from datetime import datetime
from typing import Any, Dict, Optional


class FieldMatcher(ABC):
    """Base class for field matching strategies."""
    
    @abstractmethod
    def match(self, gt_value: str, pred_value: str) -> bool:
        """Compare ground truth and predicted values."""
        pass


class ExactMatcher(FieldMatcher):
    """Exact string matching."""
    
    def match(self, gt_value: str, pred_value: str) -> bool:
        return gt_value == pred_value


class DateMatcher(FieldMatcher):
    """Date format aware matching."""
    
    def __init__(self, formats: list[str] = None):
        self.formats = formats or ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]
    
    def match(self, gt_value: str, pred_value: str) -> bool:
        try:
            for fmt in self.formats:
                try:
                    gt_date = datetime.strptime(gt_value, fmt)
                    pred_date = datetime.strptime(pred_value, fmt)
                    return gt_date == pred_date
                except ValueError:
                    continue
            return False
        except Exception:
            return gt_value == pred_value


class NumericMatcher(FieldMatcher):
    """Numeric value matching with optional tolerance."""
    
    def __init__(self, tolerance: float = 0.0):
        self.tolerance = tolerance
    
    def match(self, gt_value: str, pred_value: str) -> bool:
        try:
            # 133,33 should be equal to 13333 
            gt_num = float(str(gt_value).replace(',', ''))
            pred_num = float(str(pred_value).replace(',', ''))
            return abs(gt_num - pred_num) <= self.tolerance
        except (ValueError, TypeError):
            return gt_value == pred_value


class CaseInsensitiveMatcher(FieldMatcher):
    """Case insensitive string matching."""
    
    def match(self, gt_value: str, pred_value: str) -> bool:
        return str(gt_value).lower() == str(pred_value).lower()


class CurrencyMatcher(FieldMatcher):
    """Currency value matching with optional tolerance.
    
    Supports formats like:
    - "1,234.56 USD"
    - "1.234,56 EUR"
    - "1234.56USD"
    - "USD 1,234.56"
    - "1,234.56"
    """

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
        self.currency_pattern = r'([0-9,.]+)\s*([A-Z]{3})?|([A-Z]{3})?\s*([0-9,.]+)'

    def _extract_amount_and_currency(self, value: str) -> tuple[float, str]:
        """Extract numeric amount and currency code from string.
        
        Args:
            value: String containing amount and optional currency code
            
        Returns:
            Tuple of (amount, currency_code)
        """
        if not value or value == "N/A":
            return 0.0, ""
            
        value = str(value).strip()
        match = re.search(self.currency_pattern, value)
        if not match:
            return 0.0, ""
            
        # Get amount and currency from either format
        amount_str = match.group(1) or match.group(4)
        currency = (match.group(2) or match.group(3) or "").strip()
        
        # Clean amount string
        amount_str = amount_str.replace(' ', '')
        
        # Handle different decimal/thousand separators
        if ',' in amount_str and '.' in amount_str:
            if amount_str.find(',') < amount_str.find('.'):
                # Format: 1,234.56
                amount_str = amount_str.replace(',', '')
            else:
                # Format: 1.234,56
                amount_str = amount_str.replace('.', '').replace(',', '.')
        elif ',' in amount_str:
            # Determine if comma is decimal or thousand separator
            parts = amount_str.split(',')
            if len(parts[-1]) == 2 and len(parts) <= 2:
                # Likely decimal: 1234,56
                amount_str = amount_str.replace(',', '.')
            else:
                # Likely thousands: 1,234
                amount_str = amount_str.replace(',', '')
                
        try:
            amount = float(amount_str)
        except ValueError:
            return 0.0, ""
            
        return amount, currency

    def match(self, gt_value: str, pred_value: str) -> bool:
        """Compare currency values, optionally considering currency codes."""
        try:
            gt_amount, gt_currency = self._extract_amount_and_currency(gt_value)
            pred_amount, pred_currency = self._extract_amount_and_currency(pred_value)
            
            # If currency codes are present, they must match
            if gt_currency and pred_currency and gt_currency != pred_currency:
                return False
                
            # Compare amounts within tolerance
            return abs(gt_amount - pred_amount) <= self.tolerance
            
        except Exception:
            # Fall back to exact string comparison if parsing fails
            return gt_value == pred_value
