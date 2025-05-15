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

import yaml
from pprint import pprint
import json

def print_yaml_structure(data, indent=0):
    """Pretty print YAML structure with proper indentation."""
    indent_str = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent_str}{key}:")
            print_yaml_structure(value, indent + 1)
    else:
        # 使用json.dumps让字符串格式化更美观
        print(f"{indent_str}{json.dumps(data, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    with open("./src/openllm_ocr_annotator/config/prompt_templates.yaml", "r") as f:
        data = yaml.safe_load(f)
        print("\n=== Pretty Print using custom function ===")
        print_yaml_structure(data)
        
        print("\n=== Pretty Print using pprint ===")
        pprint(data, indent=2, width=80)
        
        print("\n=== Pretty Print using JSON ===")
        print(json.dumps(data, indent=2, ensure_ascii=False))
