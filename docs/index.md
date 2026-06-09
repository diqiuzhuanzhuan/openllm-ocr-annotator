# OpenLLM OCR Annotator

OpenLLM OCR Annotator is an OCR dataset annotation tool powered by multimodal large language models. It extracts structured fields from document images with multiple vision models and improves annotation quality through voting and evaluation.

## Features

- Supports OpenAI, Claude, Gemini, and LiteLLM providers
- Processes document images in parallel
- Provides majority and weighted voting across models
- Supports repeated sampling and field-level accuracy evaluation
- Exports JSON, JSONL, TSV, and Hugging Face datasets
- Provides manual sampling review through Streamlit

## Quick Start

```bash
just install
python apps/app.py --config examples/config.yaml
```

Run the complete quality-assurance suite:

```bash
just qa
```
