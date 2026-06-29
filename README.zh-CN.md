# OpenLLM OCR Annotator

[English](README.md) | [简体中文](README.zh-CN.md)

**OpenLLM OCR Annotator** 是一款多模态标注工具，旨在通过多种大语言模型（LLM）API 辅助完成 OCR（光学字符识别）标注。它支持生成多种格式的标注结果，帮助用户更便捷地构建图像文本识别与理解任务所需的数据集。

## ✨ 功能特性

- 🔌 **支持多种 LLM API**：兼容 OpenAI、Claude、Gemini、Qwen、Mistral、Grok 等。
- 🖼️ **多模态输入**：处理图像与文本组合输入，实现上下文感知标注。
- 📤 **灵活输出**：支持导出 JSON、YAML、纯文本等多种格式的标注结果。
- 🤗 **创建 Hugging Face 格式的数据集**：通过 Hydra 配置组灵活定制数据集生成。
- 📊 **评估**：输出字段级准确率和文档级准确率。
- ⚙️ **轻量化**：基于 Python 构建，依赖精简。
- 🌍 **开源**：欢迎参与贡献！

## 📦 安装

### ✅ Python 环境

本项目需要 **Python 3.13.2**。推荐使用 [uv](https://github.com/astral-sh/uv) 管理环境和依赖。

### 1. 克隆仓库

```bash
git clone https://github.com/diqiuzhuanzhuan/openllm-ocr-annotator.git
cd openllm-ocr-annotator
```

### 2. 创建环境并安装依赖

```bash
uv sync --group dev
```

复制环境变量模板，并配置需要使用的模型服务商：

```bash
cp .env.example .env
```

`apps/app.py` 会自动加载 `.env`。

## 🚀 快速开始

运行以下命令启动标注工具：

```bash
python apps/app.py
```

### 使用示例

标注入口现在使用 [Hydra](https://hydra.cc/) 配置组。默认入口会加载 `configs/config.yaml`，通常不再需要传入 `--config` 参数。

#### 基础用法

使用默认配置运行：

```bash
python apps/app.py
```

从命令行覆盖配置项：

```bash
python apps/app.py task.max_files=10 task.input_dir=./data/images
python apps/app.py task.output_dir=./data/outputs dataset=foreign_trade
```

只打印最终解析后的配置，不执行标注：

```bash
python apps/app.py --cfg job
```

#### 配置结构

Hydra 会从 `configs/config.yaml` 组合运行时配置：

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

任务配置位于 `configs/task/foreign_trade.yaml`：

```yaml
task_id: foreign_trade_20250519
input_dir: "./data/images"
output_dir: "./data/outputs"
max_files: 20
num_samples: 1
```

标注器配置位于 `configs/annotators/foreign_trade_default.yaml`，并通过 `annotators@task` 合并到 `task.annotators`：

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

投票和数据集配置分别放在独立配置组中：

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

`prompt_templates.yaml` 仍然通过 `prompt_path` 配置：

```yaml
openai:
  vision_extraction:
    system: |
      You are an expert in foreign trade document analysis. Your task is to extract key information
      from Chinese foreign trade documents with high precision.

    user: |
      Analyze this foreign trade document and extract the following specific fields:......
```

**注意**：所有标注结果都会存储在 `{task.output_dir}/{annotator.name}/{annotator.model}` 中。可以在 `configs/annotators/*.yaml` 中添加多个标注器，并通过命令行切换配置组。

#### 验证标注结果（抽样并检查标注）

```bash
streamlit run apps/streamlit_viewer.py
```

运行后可以查看标注准确率，并通过 Web 界面浏览标注结果。

## 开发

安装 [just](https://github.com/casey/just)，然后使用项目提供的命令：

```bash
just install       # 安装项目及开发依赖
just build         # 构建源码包和 wheel 包
just docs          # 构建 Zensical 文档站点
just docs-serve    # 启动支持实时刷新的文档预览
just test          # 运行全部测试
just test -k voter # 向 pytest 传递额外参数
just lint          # 运行 Ruff 代码检查
just fmt           # 格式化 Python 文件
just fmt-check     # 检查格式但不修改文件
just check         # 运行代码检查、格式检查和测试
just pre-commit    # 运行全部 pre-commit hooks
```

🔧 即将推出：Web 界面 / 通过 GitHub Actions Pages 发布的演示。

🤖 支持的模型服务商

- [x] OpenAI（GPT-4 / GPT-3.5）
- [x] Claude（Anthropic）
- [x] Gemini（Google）
- [x] Qwen（Alibaba）
- [x] Mistral
- [x] Grok（xAI / Elon Musk）
- [x] Litellm

📂 输出格式

- [x] JSON
- [ ] YAML
- [ ] TSV（即将支持）
- [ ] 纯文本
- [ ] XML（即将支持）
- [ ] CSV（即将支持）

## 📄 许可证

本项目采用 MIT 许可证，详情请参阅 LICENSE。

## 🧑‍💻 作者

- Loong Ma
  * 📫 diqiuzhuanzhuan@gmail.com
  * 🌐 GitHub：@diqiuzhuanzhuan

## 🙌 参与贡献

欢迎提交 Issue、功能建议或 Pull Request！

## ⭐ Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=diqiuzhuanzhuan/openllm-ocr-annotator&type=Date)](https://www.star-history.com/#diqiuzhuanzhuan/openllm-ocr-annotator&Date)
