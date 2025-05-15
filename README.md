# OpenLLM OCR Annotator

**OpenLLM OCR Annotator** is a multimodal annotation tool designed to assist with OCR (Optical Character Recognition) labeling using multiple Large Language Model (LLM) APIs. It supports generating annotations in various formats, making it easier to build datasets for tasks involving text recognition and understanding in images.

## ✨ Features

- 🔌 **Supports Multiple LLM APIs**: Compatible with OpenAI, Claude, Gemini, Qwen, Mistral, Grok, etc.
- 🖼️ **Multimodal Input**: Process image + text pairs for richer context-aware annotation.
- 📤 **Flexible Output**: Export annotations in multiple formats (JSON, YAML, plain text, etc.).
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
python app.py
```

🔧 Coming soon: Web-based UI / Demo via GitHub Actions page.

🤖 Supported Models
- [ ] OpenAI (GPT-4 / GPT-3.5)
- [ ] Claude (Anthropic)
- [ ] Gemini (Google)
- [ ] Qwen (Alibaba)
- [ ] Mistral (Coming soon)
- [ ] Grok (xAI / Elon Musk)

📂 Output Formats
- [ ] JSON
- [ ] YAML
- [ ] TSV
- [ ] Plain text
- [ ] XML (Coming soon)
- [ ] CSV (Coming soon)


## 📄 License

This project is licensed under the MIT License. See LICENSE for details.


## 🧑‍💻 Author

Loong Ma
📫 diqiuzhuanzhuan@gmail.com
🌐 GitHub: @diqiuzhuanzhuan


## 🙌 Contributions

Feel free to submit issues, feature requests, or pull requests. All contributions are welcome!

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=diqiuzhuanzhuan/openllm-ocr-annotator&type=Date)](https://star-history.com/#diqiuzhuanzhuan/openllm-ocr-annotator&Date)
