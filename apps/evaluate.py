# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT


import argparse
from src.openllm_ocr_annotator.evaluators.field_evaluator import FieldEvaluator
from utils.field_matcher import DateMatcher, CurrencyMatcher


def main():
    # groud_trueth_dir should be a directory containing JSON files, such as: image1.json, image2.json
    # predictions_dir should be a directory containing JSON files, such as: image1.json, image2.json
    # The JSON files should contain the same structure as the ground truth files
    # {"result" : {"fields": [{"field_name": "document_number", "value": "DR-20230316B", "confidence": 1.0}, ...]}}
    # "reuslt", "fields", "field_name", "value" are all required fields
    # If you have a field that is not in the ground truth, it will be ignored
    # Here is an example of the JSON file structure:
    """
    {
        "result": {
            "fields": [
            {
                "field_name": "document_number",
                "value": "0230316B",
                "confidence": 1.0
            },
            {
                "field_name": "contract_date",
                "value": "2023-03-28",
                "confidence": 1.0
            },
            {
                "field_name": "buyer_name",
                "value": "FOSFOROS",
                "confidence": 1.0
            },
            {
                "field_name": "buyer_country",
                "value": "N/A",
                "confidence": 1.0
            },
            {
                "field_name": "transaction_amount",
                "value": "85,600.00 USD",
                "confidence": 0.7888888888888889
            }
            ]
        }
    }

    """
    parser = argparse.ArgumentParser(description="Evaluate OCR annotation results")
    parser.add_argument(
        "--ground-truth", required=True, help="Directory containing ground truth files"
    )
    parser.add_argument(
        "--predictions", required=True, help="Directory containing prediction files"
    )
    parser.add_argument("--output", help="Output file for evaluation report")

    args = parser.parse_args()

    field_mathers = {
        "contract_date": DateMatcher(),
        "transaction_amount": CurrencyMatcher(),
    }

    evaluator = FieldEvaluator(
        args.ground_truth, args.predictions, field_matchers=field_mathers
    )
    results = evaluator.evaluate_batch()

    # Generate and save report
    report = evaluator.generate_report(results, args.output)
    print(report)


if __name__ == "__main__":
    main()
