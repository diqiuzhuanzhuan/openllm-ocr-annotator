# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

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
