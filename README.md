# OpenLLM OCR Annotator

[English](README.md) | [简体中文](README.zh-CN.md)

**OpenLLM OCR Annotator** is a multimodal annotation tool designed to assist with OCR (Optical Character Recognition) labeling using multiple Large Language Model (LLM) APIs. It supports generating annotations in various formats, making it easier to build datasets for tasks involving text recognition and understanding in images.

## ✨ Features

- 🔌 **Supports Multiple LLM APIs**: Compatible with OpenAI, Claude, Gemini, Qwen, Mistral, Grok, etc.
- 🖼️ **Multimodal Input**: Process image + text pairs for richer context-aware annotation.
- 📤 **Flexible Output**: Export annotations in multiple formats (JSON, YAML, plain text, etc.).
- 🤗 **Create huggingface format dataset**: Hydra configuration groups make dataset generation easy to customize.
- 📊 **Evaluate**: Output field-level accuracy and document-level accuracy.
- ⚙️ **Lightweight**: Built in Python with minimal dependencies.
- 🌍 **Open Source**: Contributions welcome!


## 📦 Installation

### ✅ Python Environment

This project requires **Python 3.13.2**. We recommend using [uv](https://github.com/astral-sh/uv) for environment and dependency management.

### 1. Clone the repository

```bash
git clone https://github.com/diqiuzhuanzhuan/openllm-ocr-annotator.git
cd openllm-ocr-annotator
```
### 2. Create the environment and install dependencies

```bash
uv sync --group dev
```

Copy the environment template and configure the providers you use:

```bash
cp .env.example .env
```

`apps/app.py` loads `.env` automatically.

## 🚀 Quick Start

You can run the annotator using:
```bash
python apps/app.py
```

### Usage Examples

The annotator now uses [Hydra](https://hydra.cc/) configuration groups. The default entry point loads `configs/config.yaml`, so most runs do not need a `--config` argument.

#### Basic Usage

Run with the default configuration:

```bash
python apps/app.py
```

Override values from the command line:

```bash
python apps/app.py task.max_files=10 task.input_dir=./data/images
python apps/app.py task.output_dir=./data/outputs dataset=foreign_trade
```

Print the resolved job configuration without running annotation:

```bash
python apps/app.py --cfg job
```

#### Configuration Layout

Hydra composes the runtime configuration from `configs/config.yaml`:

```yaml
defaults:
  - task: foreign_trade
  - annotators@task: foreign_trade_default
  - ensemble@task: weighted_vote
  - dataset@task: foreign_trade
  - _self_

version: "1.0"

hydra:
  job:
    chdir: false
```

The task settings live in `configs/task/foreign_trade.yaml`:

```yaml
task_id: foreign_trade_20250519
input_dir: "./data/images"
output_dir: "./data/outputs"
max_files: 20
num_samples: 1
```

Annotators live in `configs/annotators/foreign_trade_default.yaml` and are merged under `task.annotators` by `annotators@task`:

```yaml
annotators:
  - name: "gpt"
    type: curator
    task: "vision_extraction"
    provider:
      model_name: "openai/gpt-5"
      backend: "litellm"
      backend_params:
        require_all_responses: false
        max_tokens_per_minute: 100000000
      generation_params:
        max_tokens: 4096
    weight: 1.0
    output_format: json
    enabled: true
    prompt_path: "./examples/prompt_templates.yaml"
```

Ensemble and dataset options are configured separately:

```yaml
# configs/ensemble/weighted_vote.yaml
ensemble:
  enabled: true
  method: "weighted_vote"
  min_confidence: 0.5
  agreement_threshold: 0.6
  output_format: json
```

```yaml
# configs/dataset/foreign_trade.yaml
dataset:
  enabled: true
  name: "foreign_trade_20250519"
  version: "1.0"
  description: "Dataset for foreign trade document analysis."
  format: json
  output_dir: "./datasets/foreign_trade_20250519"
  split_ratio: 0.9
  num_samples: -1
```

`prompt_templates.yaml` is still configured through `prompt_path`:

```yaml
curator:
  vision_extraction:
    system: |
      You are an expert in foreign trade document analysis. Your task is to extract key information
      from Chinese foreign trade documents with high precision.

    user: |
      Analyze this foreign trade document and extract the following specific fields:......
```

**Note**: annotation results are stored under `{task.output_dir}/{annotator.name}/{annotator.model}`. You can add more annotators to `configs/annotators/*.yaml` and switch between them from the command line.

#### verify the annotations (sample n samples and verify the annotations)
```python
streamlit run apps/streamlit_viewer.py
```
You will get the accuracy of the annotations, and you can also view the annotations in a web-based UI.

## Development

Install [just](https://github.com/casey/just) and use the project recipes:

```bash
just install       # Install project and development dependencies
just build         # Build source and wheel distributions
just docs          # Build the Zensical documentation site
just docs-serve    # Preview documentation with live reload
just test          # Run all tests
just test -k voter # Pass additional arguments to pytest
just lint          # Run Ruff lint checks
just fmt           # Format Python files
just fmt-check     # Check formatting without modifying files
just check         # Run lint, format checks, and tests
just pre-commit    # Run all pre-commit hooks
```

🔧 Coming soon: Web-based UI / Demo via GitHub Actions page.

🤖 Supported Model Providers
- [x] OpenAI (GPT-4 / GPT-3.5)
- [x] Claude (Anthropic)
- [x] Gemini (Google)
- [x] Qwen (Alibaba)
- [x] Mistral
- [x] Grok (xAI / Elon Musk)
- [x] Litellm

📂 Output Formats
- [x] JSON
- [ ] YAML
- [ ] TSV (Comming soon)
- [ ] Plain text
- [ ] XML (Coming soon)
- [ ] CSV (Coming soon)


## 📄 License

This project is licensed under the MIT License. See LICENSE for details.


## 🧑‍💻 Author

- Loong Ma
  * 📫 diqiuzhuanzhuan@gmail.com
  * 🌐 GitHub: @diqiuzhuanzhuan


## 🙌 Contributions

Feel free to submit issues, feature requests, or pull requests. All contributions are welcome!

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=diqiuzhuanzhuan/openllm-ocr-annotator&type=Date)](https://www.star-history.com/#diqiuzhuanzhuan/openllm-ocr-annotator&Date)
