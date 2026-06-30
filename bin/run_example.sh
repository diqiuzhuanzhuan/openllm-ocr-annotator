#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Examples:
#   bin/run_example.sh task.max_files=10 task.input_dir=./data/images
#   bin/run_example.sh task.annotators.0.type=curator task.annotators.0.task=vision_extraction
#
# Hydra list items are addressed by index, so the first annotator is
# task.annotators.0, the second annotator is task.annotators.1, and so on.

python apps/app.py \
    task.input_dir=./data/images \
    task.max_files=50 \
    task.annotators.0.type=curator \
    task.annotators.0.task=vision_extraction \
    task.annotators.0.prompt_path=./examples/prompt_templates.yaml \
    task.annotators.1.type=curator \
    task.annotators.1.task=vision_extraction \
    task.annotators.1.prompt_path=./examples/prompt_templates.yaml
