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

import os
from annotators.openai import OpenAIAnnotator
from utils.formatter import save_as_jsonl, save_as_tsv

annotator = OpenAIAnnotator(config_path="config/model_api.yaml")

input_dir = "data/images"
output_dir = "data/outputs"

for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    fpath = os.path.join(input_dir, fname)
    result = annotator.annotate(fpath)

    save_as_jsonl(result, os.path.join(output_dir, "jsonl", fname + ".jsonl"))
    save_as_tsv(result, os.path.join(output_dir, "tsv", fname + ".tsv"))