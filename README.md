# OpenLLM OCR Annotator

**OpenLLM OCR Annotator** is a multimodal annotation tool designed to assist with OCR (Optical Character Recognition) labeling using multiple Large Language Model (LLM) APIs. It supports generating annotations in various formats, making it easier to build datasets for tasks involving text recognition and understanding in images.

## ✨ Features

- 🔌 **Supports Multiple LLM APIs**: Compatible with OpenAI, Claude, Gemini, Qwen, Mistral, Grok, etc.
- 🖼️ **Multimodal Input**: Process image + text pairs for richer context-aware annotation.
- 📤 **Flexible Output**: Export annotations in multiple formats (JSON, YAML, plain text, etc.).
- 🤗 **Create huggingface format dataset**: Only a few lines of configuration in config.yaml.
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
### 2. Create and activate a virtual environment

```bash
uv venv .venv
source .venv/bin/activate
```
### 3. Install dependencies
```bash
uv pip install .
```

## 🚀 Quick Start

You can run the annotator using:
```bash
python apps/app.py
```

### Usage Examples
#### Basic Usage
Create a 'config.yaml' file in the 'examples' directory:
```yaml
name: my_annotator
model: gpt-4-vision-preview  # or gemini-pro-vision
api_key: your_api_key_here
task: vision_extraction
type: openai
base_url: 'http://127.0.0.1:8879/v1'
enabled: true                # Disabled by default
max_tokens: 1000
weight: 1
output_format: json
temperature: null
prompt_path: "./examples/prompt_templates.yaml"
```
**Actually, refering to config.yaml provided in the 'examples' directory is the best choice.**
#### basic useage
```python
python apps/app.py --config examples/config.yaml
```


🔧 Coming soon: Web-based UI / Demo via GitHub Actions page.

🤖 Supported Models
- [x] OpenAI (GPT-4 / GPT-3.5)
- [ ] Claude (Anthropic, Comming soon)
- [x] Gemini (Google)
- [ ] Qwen (Alibaba)
- [ ] Mistral (Coming soon)
- [ ] Grok (xAI / Elon Musk)

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
