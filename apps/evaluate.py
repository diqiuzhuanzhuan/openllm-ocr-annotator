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


import argparse
from pathlib import Path
from src.openllm_ocr_annotator.evaluators.field_evaluator import FieldEvaluator

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
    parser.add_argument("--ground-truth", required=True, help="Directory containing ground truth files")
    parser.add_argument("--predictions", required=True, help="Directory containing prediction files")
    parser.add_argument("--output", help="Output file for evaluation report")
    
    args = parser.parse_args()
    
    evaluator = FieldEvaluator(args.ground_truth, args.predictions)
    results = evaluator.evaluate_batch()
    
    # Generate and save report
    report = evaluator.generate_report(results, args.output)
    print(report)

if __name__ == "__main__":
    main()