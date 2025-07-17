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
version: "1.0"
task:
  # Basic Configuration
  # -----------------------------------------------------------------------------
  task_id: mytask
  input_dir: "./data/images"           # Source image directory
  output_dir: "./data/outputs"         # Output directory for annotations
  max_files: -1                       # Set as 10 when to test overall process.

annotators:
  - name: my_annotator
    model: gpt-4-vision-preview  # or gemini-pro-vision
    api_key: your_api_key_here
    task: vision_extraction
    type: openai                 # Use api compatible with OpenAI
    base_url: 'http://127.0.0.1:8879/v1'   # If you set up your own OpenAI compatible API server
    enabled: true                # Disable this annotator by setting to false
    max_retries: 3
    max_tokens: 1000
    weight: 1
    output_format: json
    temperature: null
    prompt_path: "./examples/prompt_templates.yaml"
```

prompt_templates.yaml:

```yaml
openai:  # must be the same as the 'type' field in the annotator configuration
  vision_extraction: # must be the same as the 'task' field in the annotator configuration
    system: |
      You are an expert in foreign trade document analysis. Your task is to extract key information
      from Chinese foreign trade documents with high precision. Pay special attention to:
      1. Document identifiers and numbers
      2. Dates in standard formats
      3. Company names and addresses
      4. Transaction amounts and currencies
      5. Geographic information

    user: |
      Analyze this foreign trade document and extract the following specific fields:......

```
**Note**: All annotation results will be stored in '{task.output_dir}/{annotator.name}/{annotator.model}'. So you can set up many annotators.

**Actually, refering to config.yaml provided in the 'examples' directory is the best choice.**
#### basic useage
```python
python apps/app.py --config examples/config.yaml
```

#### verify the annotations (sample n samples and verify the annotations)
```python
streamlit run apps/stramlit_viewer.py
```
You will get the accuracy of the annotations, and you can also view the annotations in a web-based UI.

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
