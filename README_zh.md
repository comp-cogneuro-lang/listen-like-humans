# 机器是否像人类一样聆听？端到端 ASR 中音系竞争的时间动态基准

**🌐 语言：** [English](README.md) | 简体中文

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Interactive results](https://img.shields.io/badge/results-interactive%20demo-6aa8ff.svg)](https://comp-cogneuro-lang.github.io/listen-like-humans/)

> 🔎 **交互式网站：** [comp-cogneuro-lang.github.io/listen-like-humans](https://comp-cogneuro-lang.github.io/listen-like-humans/) —— 可叠加对比模型与人类的词汇激活轨迹、按模型浏览结果，或者按照指标 RMSE/MAE 进行排序。

论文 **《Do Machines Listen Like Humans? A Temporal Benchmark for Phonological Competition in End-to-End ASR》**（Interspeech 2026）的官方实现。本仓库提供代码与数据，用于评估自动语音识别（ASR）模型是否像人类一样以增量方式处理语音，并表现出与人类相似的词汇竞争动态。

## 概述

人类的语音识别是增量式的：随着语音的展开，听者会持续地激活并抑制相互竞争的候选词。本基准定量地比较了 ASR 模型中词汇激活的时间进程与人类眼动追踪数据（来自视觉世界范式（Visual World Paradigm, VWP）。我们在时间维度上检测模型的内部状态，并测量目标词、同起首竞争词（cohort，前两个音素相同的词汇竞争者。比如becker,beetle）、押韵竞争词（rhyme，词首不同但词尾相同， 比如hat，cat）以及无关词的激活曲线。所得到的轨迹再与人类注视比例（fixation proportion）逐点比较，使用 RMSE 与 MAE 作为度量。注意 RMSE 与 MAE 本身无法作为判断标准，竞争轨迹才是：先出现早期的 cohort 竞争，随后出现较晚的 rhyme 激活。

**核心发现：** 因果（causal）架构（LSTM、causal CNN、causal RCNN）能够复现人类的标志性模式：先出现早期的 cohort 竞争，随后出现较晚的 rhyme 激活；而带有前瞻（look-ahead）的非因果模型（BiLSTM、Transformer、ConvTransformer）以及大型预训练 ASR 模型（wav2vec 2.0、HuBERT、Whisper）尽管转录准确率更高，却无法捕捉这些时间动态。这些结果提醒我们：**不能仅凭高准确率就断言某个模型"类脑"，而不去评估其时间动态。**

## 数据集

我们使用了一个设计过的词表，包含 1,533 个无屈折变化的英文单词（1–16 个音素）。音频由六位合成说话人（Apple "Say" 应用）和一位真人说话人录制，共得到 7 × 1,533 = 10,731 条语音。每个单词都配有一个中心化的 300 维 word2vec 词向量（fasttext 300d english）作为语义目标。

人类注视数据来源于 Allopenna 等人（1998）的研究，并被处理为按时间归一化的目标、cohort、rhyme 与无关条件的比例。人类注视数据经原作者（Allopenna 等，1998）许可使用。

### 下载数据

人类注视数据（`notebooks/INPUT/amt_human_mean.csv`）已包含在本仓库中。音频（`*.wav`，16 kHz）**未**包含——请从 [Zenodo](https://zenodo.org/records/20564345) 下载 `dataset.tar.gz`，并在仓库根目录解压：

```bash
tar -xzf dataset.tar.gz   # 解压到 dataset/en/<speaker>/<word>.wav
```

频谱图会在首次运行时计算并预取到内存中；一个小型的类别字典缓存会自动写入 `cache/`。

## 环境依赖

代码在 **Python 3.10+** 与 PyTorch 下开发；训练推荐使用支持 CUDA 的 GPU。安装依赖：

```bash
pip install -r requirements.txt
```

说明：
- 训练默认会将日志记录到 [Weights & Biases](https://wandb.ai)——设置 `USE_WANDB=0` 可关闭（无需账号）。
- 评估基础模型（wav2vec 2.0、HuBERT、Whisper）会在首次运行时从 Hugging Face Hub 下载预训练权重，因此这些任务需要联网（或为离线节点预先填充 `HF_HOME` 缓存）。
- 生成图表的 notebook 还需要 Jupyter（`pip install jupyter`）。

## 模型库（Model Zoo）

我们评估了一系列**因果**（增量式）与**非因果**（能够得到来自未来时间步的数据）变体架构，参数量相近（约 1.6–1.9M）。所有模型均在孤立词任务上从零开始训练，使用最终隐藏状态与目标 word2vec 词向量之间的 MSE 损失。

| 模型                 | 类型        | 描述 |
|----------------------|-------------|-------------|
| Baseline LSTM        | 因果        | 单层单向 LSTM |
| 2L-LSTM              | 因果        | 双层单向 LSTM |
| Causal-CNN           | 因果        | 带因果填充的一维卷积 |
| Causal-RCNN          | 因果        | 一维 CNN + 单向 LSTM |
| Causal-Transformer   | 因果        | 带因果自注意力的 Transformer |
| 2L-BiLSTM            | 非因果      | 双向 LSTM（完整上下文） |
| RCNN                 | 非因果      | 非因果 CNN（25 帧前瞻）+ LSTM |
| CNN                  | 非因果      | 标准一维 CNN（完整上下文） |
| Transformer          | 非因果      | 完整双向自注意力 |
| ConvTransformer      | 非因果      | Conformer 风格（完整上下文） |

### 因果性检查
用于检查因果性的脚本位于 `misc/causal-validation` 文件夹下。

### 参数量
所有模型的参数量可通过 `misc/print_model_parameters.py` 获得。

我们同时评估了无需微调的预训练基础模型：`wav2vec2`、`hubert` 与 `whisper`。对于这些模型，我们从 CTC 对齐路径或注意力加权的 token 概率中推导出词激活概率，并通过 Luce 选择规则（Luce's choice rule）将其转换为竞争词激活分数。

## 训练与测试

### 训练模型

```bash
bash train.sh <experiment_key>
```

`<experiment_key>` 是下表中的某个键名，**不是** Model Zoo 中的显示名称。不带参数运行 `train.sh` 可打印完整列表。

| `<experiment_key>` | 模型（见 Model Zoo） | 类型 |
|--------------------|-----------------------|------|
| `baseline`            | Baseline LSTM        | 因果 |
| `causal-2lstm`        | 2L-LSTM              | 因果 |
| `causal-cnn`          | Causal-CNN           | 因果 |
| `causal-rcnn`         | Causal-RCNN          | 因果 |
| `causal-trans`        | Causal-Transformer   | 因果 |
| `causal-ctrans`       | Causal ConvTransformer | 因果 |
| `noncausal-2lstm`     | 2L-BiLSTM            | 非因果 |
| `noncausal-cnn`       | CNN                  | 非因果 |
| `noncausal-rcnn`      | RCNN                 | 非因果 |
| `noncausal-trans`     | Transformer          | 非因果 |
| `noncausal-convtrans` | ConvTransformer      | 非因果 |

检查点会写入 `experiments/<config_name>/<run_name>/pretraining/model_state_<epoch>.pth`。

### 测试模型并计算音系竞争

要评估已训练的模型并计算音系竞争轨迹（目标、cohort、rhyme 与无关词激活），传入同一个键名以及一个或多个要评估的检查点 epoch：

```bash
bash test.sh <experiment_key> <epoch1> [epoch2 ...]
```

不带 epoch 运行 `bash test.sh <experiment_key>` 会列出该模型可用的检查点。该命令调用 [`analysis/_comp_competition_batch.py`](analysis/_comp_competition_batch.py)，计算各竞争类型的逐时刻激活轨迹并写入 `experiments/<config_name>/<run_name>/training/competition.csv`（同时给出词识别准确率）。相对于人类 VWP 注视数据的 RMSE/MAE 以及最终的对比图，则由 notebook 从该 CSV 生成（见**复现论文图表**）。

## 评估基础 ASR 模型

我们还评估了无需微调的预训练基础模型（wav2vec 2.0、HuBERT、Whisper）。评估脚本位于 `pretrained_models/`：

### wav2vec 2.0 与 HuBERT

使用 `eval_wav2vec2.py` 评估 wav2vec 2.0 与 HuBERT——脚本通过 `Auto*` 类加载模型，因此同一段代码可处理两种架构。通过环境变量 `MODEL_NAME` 与 `OUTPUT_DIR` 选择模型与输出目录（两者都有合理的默认值）：

```bash
# wav2vec 2.0（默认）-> 写入 experiments/wav2vec2/competition.csv
python pretrained_models/eval_wav2vec2.py

# HuBERT -> 写入 experiments/hubert/competition.csv
MODEL_NAME=facebook/hubert-large-ls960-ft OUTPUT_DIR=experiments/hubert \
    python pretrained_models/eval_wav2vec2.py
```

### Whisper

使用 `eval_whisper.py` 评估 Whisper 模型：

```bash
python pretrained_models/eval_whisper.py
```

**关于 Nemotron 的说明：** 我们也得到了 Nemotron 模型的评估结果；但出于论文中所述原因，我们决定不在论文中报告这些结果。

## 结果

我们的主要结果表明，**因果模型**比**非因果**模型以及许多现成的预训练 ASR 模型更好地匹配人类 VWP 动态（RMSE/MAE 更低）。这些结果提醒我们：**不能仅凭高准确率就断言某个模型"类脑"，而不去评估其时间动态。** 各模型的激活轨迹写入 `experiments/<model>/competition.csv`，最终的度量与图表则由 `notebooks/` 中的 notebook 从这些 CSV 生成（见下文）。

## 复现论文图表

详细的分析与图表生成 notebook 位于 `notebooks/`：

- **度量指标：** 见 [notebooks/calculate_RMSE_MAE.ipynb](notebooks/calculate_RMSE_MAE.ipynb)，用于计算模型轨迹与人类数据之间的 RMSE 和 MAE。

- **图 2 与图 4：** 见 [notebooks/Fig2_4.ipynb](notebooks/Fig2_4.ipynb)，可视化激活轨迹，将人类 VWP 注视与模型预测对比。这些图展示了特征性的时间动态：因果模型中先出现早期 cohort 竞争，随后出现 rhyme 激活。

- **图 3：** 见 [notebooks/Fig3.ipynb](notebooks/Fig3.ipynb)，对不同模型架构内部各层表示进行音素解码器分析。

## 引用

如果您使用了本基准或代码，请引用该论文。临时 bibtex（将在论文被接收后更新）：

```
@misc{htp2025,
  title={Do Machines Listen Like Humans? A Temporal Benchmark for Phonological Competition in End-to-End ASR},
  author={Anonymous},
  booktitle={Interspeech 2026},
  year={2026}
}
```

## 许可证

本项目以 MIT 许可证发布。人类注视数据经原作者（Allopenna 等，1998）许可使用。

## 联系方式

如有问题或疑问，请提交 GitHub issue，或联系作者（联系方式见论文）。

---

**致谢**

本工作建立在包括 EARSHOT 模型与视觉世界范式数据（Allopenna 等，1998）在内的既有资源之上。我们感谢这些资源的创建者。
