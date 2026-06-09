# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

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
