# OpenLLM OCR Annotator

[English](README.md) | [简体中文](README.zh-CN.md)

**OpenLLM OCR Annotator** 是一款多模态标注工具，旨在通过多种大语言模型（LLM）API 辅助完成 OCR（光学字符识别）标注。它支持生成多种格式的标注结果，帮助用户更便捷地构建图像文本识别与理解任务所需的数据集。

## ✨ 功能特性

- 🔌 **支持多种 LLM API**：兼容 OpenAI、Claude、Gemini、Qwen、Mistral、Grok 等。
- 🖼️ **多模态输入**：处理图像与文本组合输入，实现上下文感知标注。
- 📤 **灵活输出**：支持导出 JSON、YAML、纯文本等多种格式的标注结果。
- 🤗 **创建 Hugging Face 格式的数据集**：只需在 `config.yaml` 中添加少量配置。
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

#### 基础配置

在 `examples` 目录中创建 `config.yaml` 文件：

```yaml
version: "1.0"
task:
  # 基础配置
  # -----------------------------------------------------------------------------
  task_id: mytask
  input_dir: "./data/images"           # 原始图像目录
  output_dir: "./data/outputs"         # 标注结果输出目录
  max_files: -1                        # 测试完整流程时可设置为 10

annotators:
  - name: my_annotator
    model: gpt-4o-mini                  # 或 gpt-4o
    api_key: your_api_key_here
    task: vision_extraction
    type: curator                       # 使用 Curator 标注器
    base_url: 'http://127.0.0.1:8879/v1' # 自建 OpenAI 兼容 API 服务的地址
    enabled: true                       # 设置为 false 可禁用此标注器
    max_retries: 3
    max_tokens: 1000
    weight: 1
    output_format: json
    temperature: null
    prompt_path: "./examples/prompt_templates.yaml"
```

`prompt_templates.yaml`：

```yaml
openai:  # 必须与标注器配置中的 'type' 字段一致
  vision_extraction: # 必须与标注器配置中的 'task' 字段一致
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

**注意**：所有标注结果都会存储在 `{task.output_dir}/{annotator.name}/{annotator.model}` 中，因此可以配置多个标注器。

**推荐直接参考 `examples` 目录中提供的 `config.yaml`。**

#### 基础用法

```bash
python apps/app.py
python apps/app.py task.max_files=10 task.input_dir=./data/images
```

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
