from openllm_ocr_annotator.utils.field_matcher import (
    ExactMatcher,
    CaseInsensitiveMatcher,
    DateMatcher,
    NumericMatcher,
    CurrencyMatcher,
)


class TestExactMatcher:
    def setup_method(self):
        self.matcher = ExactMatcher()

    def test_identical_strings_match(self):
        assert self.matcher.match("hello", "hello") is True

    def test_different_strings_no_match(self):
        assert self.matcher.match("hello", "world") is False

    def test_case_sensitive(self):
        assert self.matcher.match("Hello", "hello") is False

    def test_empty_strings_match(self):
        assert self.matcher.match("", "") is True


class TestCaseInsensitiveMatcher:
    def setup_method(self):
        self.matcher = CaseInsensitiveMatcher()

    def test_same_case_matches(self):
        assert self.matcher.match("hello", "hello") is True

    def test_different_case_matches(self):
        assert self.matcher.match("Hello", "hello") is True
        assert self.matcher.match("WORLD", "world") is True

    def test_different_values_no_match(self):
        assert self.matcher.match("foo", "bar") is False


class TestDateMatcher:
    def setup_method(self):
        self.matcher = DateMatcher()

    def test_same_date_same_format(self):
        assert self.matcher.match("2024-01-15", "2024-01-15") is True

    def test_same_date_slash_format(self):
        assert self.matcher.match("2024/01/15", "2024/01/15") is True

    def test_different_dates_no_match(self):
        assert self.matcher.match("2024-01-15", "2024-01-16") is False

    def test_invalid_date_returns_false(self):
        # DateMatcher returns False when neither value matches any known date format
        assert self.matcher.match("not-a-date", "not-a-date") is False
        assert self.matcher.match("not-a-date", "other-string") is False


class TestNumericMatcher:
    def test_equal_integers(self):
        matcher = NumericMatcher()
        assert matcher.match("42", "42") is True

    def test_equal_floats(self):
        matcher = NumericMatcher()
        assert matcher.match("3.14", "3.14") is True

    def test_within_tolerance(self):
        matcher = NumericMatcher(tolerance=0.1)
        assert matcher.match("10.0", "10.05") is True

    def test_outside_tolerance(self):
        matcher = NumericMatcher(tolerance=0.01)
        assert matcher.match("10.0", "10.5") is False

    def test_comma_separated_numbers(self):
        matcher = NumericMatcher()
        assert matcher.match("1,000", "1000") is True

    def test_non_numeric_falls_back_to_exact(self):
        matcher = NumericMatcher()
        assert matcher.match("N/A", "N/A") is True
        assert matcher.match("N/A", "unknown") is False


class TestCurrencyMatcher:
    def setup_method(self):
        self.matcher = CurrencyMatcher(tolerance=0.01)

    def test_same_amount_and_currency(self):
        assert self.matcher.match("1,234.56 USD", "1234.56 USD") is True

    def test_same_amount_no_currency(self):
        assert self.matcher.match("1234.56", "1234.56") is True

    def test_mismatched_currencies_no_match(self):
        assert self.matcher.match("100 USD", "100 EUR") is False

    def test_within_tolerance(self):
        assert self.matcher.match("100.00 USD", "100.005 USD") is True

    def test_european_format(self):
        # 1.234,56 EUR == 1234.56 EUR
        assert self.matcher.match("1.234,56 EUR", "1234.56 EUR") is True

    def test_na_value_matches_zero(self):
        # N/A is treated as 0.0
        assert self.matcher.match("N/A", "N/A") is True

    def test_amount_prefix_currency(self):
        assert self.matcher.match("USD 1234.56", "1234.56 USD") is True
