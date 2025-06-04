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

# apps/evaluate_sampling.py

import argparse
from src.openllm_ocr_annotator.evaluators.sampling_evaluator import SamplingEvaluator
from utils.field_matcher import DateMatcher, CurrencyMatcher


def main():
    parser = argparse.ArgumentParser(description="Evaluate model sampling performance")
    parser.add_argument(
        "--ground-truth", required=True, help="Directory containing ground truth files"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Base directory containing sample directories",
    )
    parser.add_argument(
        "--num-samples", type=int, default=128, help="Number of samples per image"
    )
    parser.add_argument("--output", help="Output file for evaluation report")

    args = parser.parse_args()

    # Setup field matchers
    field_matchers = {
        "contract_date": DateMatcher(),
        "transaction_amount": CurrencyMatcher(),
    }

    # Create evaluator
    evaluator = SamplingEvaluator(
        ground_truth_dir=args.ground_truth,
        predictions_base_dir=args.predictions,
        field_matchers=field_matchers,
        num_samples=args.num_samples,
    )

    # Run evaluation
    results = evaluator.evaluate_batch()

    # Generate report
    report = evaluator.generate_report(results, args.output)
    print(report)


if __name__ == "__main__":
    main()
