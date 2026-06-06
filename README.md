# Do Machines Listen Like Humans? A Temporal Benchmark for Phonological Competition in End-to-End ASR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Interactive results](https://img.shields.io/badge/results-interactive%20demo-6aa8ff.svg)](https://comp-cogneuro-lang.github.io/listen-like-humans/)

> 🔎 **Explore the results interactively:** [comp-cogneuro-lang.github.io/listen-like-humans](https://comp-cogneuro-lang.github.io/listen-like-humans/) — overlay model and human activation trajectories, browse a per-model grid, and sort the RMSE/MAE table.

Official implementation of the paper **"Do Machines Listen Like Humans? A Temporal Benchmark for Phonological Competition in End-to-End ASR"** (Interspeech 2026). This repository provides code and data to evaluate whether automatic speech recognition (ASR) models process speech incrementally with human-like lexical competition dynamics.

## Overview

Human speech recognition is incremental: listeners continuously activate and suppress competing word candidates as speech unfolds. This benchmark quantitatively compares the time course of lexical activation in ASR models against human eyetracking data from the Visual World Paradigm (VWP). We probe internal model states over time and measure activation profiles for target words, cohort competitors (same onset), rhyme competitors (different onset, same ending), and unrelated words. The resulting trajectories are compared to human fixation proportions using point-wise RMSE and MAE.

**Key finding:** Causal architectures (LSTM, causal CNN, causal RCNN) replicate the hallmark human pattern—early cohort competition followed by later rhyme activation—while non-causal models with look-ahead (BiLSTM, Transformer, ConvTransformer) and large pretrained ASR models (wav2vec 2.0, HuBERT, Whisper) fail to capture these temporal dynamics despite higher transcription accuracy.

## Dataset

We use a controlled lexicon of 1,533 uninflected English words (1–16 phonemes). Audio was recorded from six synthetic talkers (Apple "Say" app) and one human speaker, resulting in 7 × 1,533 = 10,731 utterances. Each word is paired with a centered 300‑dimensional word2vec embedding (fasttext 300d english) as the semantic target.

Human fixation data are derived from the Allopenna et al. (1998) study and processed into time-normalized proportions for target, cohort, rhyme, and unrelated conditions. These are provided in `notebooks/INPUT/amt_human_mean.csv`.

### Download the data

The human fixation data (`notebooks/INPUT/amt_human_mean.csv`) are already included in this repo. The audio (`*.wav`, 16 kHz) is **not** — download `dataset.tar.gz` from [Zenodo](https://zenodo.org/records/20564345) and unpack it at the repo root:

```bash
tar -xzf dataset.tar.gz   # populates dataset/en/<speaker>/<word>.wav
```

Spectrograms are computed and prefetched into memory on first run; a small category-dictionary cache is written to `cache/` automatically.

## Requirements

The code was developed with **Python 3.10+** and PyTorch; a CUDA GPU is recommended for training. Install dependencies with:

```bash
pip install -r requirements.txt
```

Notes:
- Training logs to [Weights & Biases](https://wandb.ai) by default — set `USE_WANDB=0` to disable it (no account required).
- Evaluating the foundation models (wav2vec 2.0, HuBERT, Whisper) downloads pretrained weights from the Hugging Face Hub on first run, so those jobs need internet access (or a pre-populated `HF_HOME` cache for offline nodes).
- The figure notebooks additionally require Jupyter (`pip install jupyter`).
 
## Model Zoo

We evaluate a range of architectures with **causal** (incremental) and **non-causal** (full-utterance) variants, all with comparable parameter counts (~1.6–1.9M). Models are trained from scratch on the isolated word task using MSE loss between the final hidden state and the target word2vec embedding.

| Model                | Type        | Description |
|----------------------|-------------|-------------|
| Baseline LSTM        | Causal      | Single-layer unidirectional LSTM |
| 2L-LSTM              | Causal      | Two-layer unidirectional LSTM |
| Causal-CNN           | Causal      | 1D convolutions with causal padding |
| Causal-RCNN          | Causal      | 1D CNN + unidirectional LSTM |
| Causal-Transformer   | Causal      | Transformer with causal self-attention |
| 2L-BiLSTM            | Non-causal  | Bidirectional LSTM (full context) |
| RCNN                 | Non-causal  | Non-causal CNN (25-frame look-ahead) + LSTM |
| CNN                  | Non-causal  | Standard 1D CNN (full context) |
| Transformer          | Non-causal  | Full bidirectional self-attention |
| ConvTransformer      | Non-causal  | Conformer-style with full context |

### Causality Check
You can find scripts used to check causality under the folder `misc/causal-validation`.

### Num of Parameter
All of the models's parameter can be obtained by `misc/print_model_parameters.py`.

We also evaluate pretrained foundation models (no fine-tuning): `wav2vec2`, `hubert`, and `whisper`. For these, we derive word activation probabilities from CTC alignment paths or attention-weighted token probabilities and transform them (via Luce's choice rule) to obtain competitor activation scores.

## Training and Testing

### Train models

```bash
bash train.sh <experiment_key>
```

`<experiment_key>` is one of the keys below, **not** the Model-Zoo display name. Run `train.sh` with no argument to print the full list.

| `<experiment_key>` | Model (see Model Zoo) | Type |
|--------------------|-----------------------|------|
| `baseline`            | Baseline LSTM        | Causal |
| `causal-2lstm`        | 2L-LSTM              | Causal |
| `causal-cnn`          | Causal-CNN           | Causal |
| `causal-rcnn`         | Causal-RCNN          | Causal |
| `causal-trans`        | Causal-Transformer   | Causal |
| `causal-ctrans`       | Causal ConvTransformer | Causal |
| `noncausal-2lstm`     | 2L-BiLSTM            | Non-causal |
| `noncausal-cnn`       | CNN                  | Non-causal |
| `noncausal-rcnn`      | RCNN                 | Non-causal |
| `noncausal-trans`     | Transformer          | Non-causal |
| `noncausal-convtrans` | ConvTransformer      | Non-causal |

Checkpoints are written to `experiments/<config_name>/<run_name>/pretraining/model_state_<epoch>.pth`.

### Test models and compute phonological competition

To evaluate a trained model and compute phonological competition trajectories (target, cohort, rhyme, and unrelated word activations), pass the same key plus one or more checkpoint epochs to evaluate:

```bash
bash test.sh <experiment_key> <epoch1> [epoch2 ...]
```

Run `bash test.sh <experiment_key>` with no epochs to list the available checkpoints for that model. This calls [`analysis/_comp_competition_batch.py`](analysis/_comp_competition_batch.py), which computes the per-competitor activation trajectories and writes them to `experiments/<config_name>/<run_name>/training/competition.csv` (plus a word-recognition accuracy). The RMSE/MAE against human VWP fixations and the final comparison figures are produced from that CSV by the notebooks (see **Reproducing Paper Figures**).

## Evaluating Foundational ASR Models

We also evaluate pretrained foundation models (wav2vec 2.0, HuBERT, Whisper) without fine-tuning. Scripts for evaluating these models are located in `pretrained_models/`:

### wav2vec 2.0 and HuBERT

Use `eval_wav2vec2.py` to evaluate both wav2vec 2.0 and HuBERT — the script loads the model with `Auto*` classes, so the same code handles either architecture. Select the model and output directory with the `MODEL_NAME` and `OUTPUT_DIR` environment variables (both have sensible defaults):

```bash
# wav2vec 2.0 (default) -> writes experiments/wav2vec2/competition.csv
python pretrained_models/eval_wav2vec2.py

# HuBERT -> writes experiments/hubert/competition.csv
MODEL_NAME=facebook/hubert-large-ls960-ft OUTPUT_DIR=experiments/hubert \
    python pretrained_models/eval_wav2vec2.py
```

### Whisper

Use `eval_whisper.py` to evaluate Whisper models:

```bash
python pretrained_models/eval_whisper.py 
```

**Note on Nemotron:** We also have evaluation results for Nemotron models; however, we decided not to report these in the paper due to the reason we mentioned in the paper.


## Results

Our main results show that **causal models** better match human VWP dynamics (lower RMSE/MAE) than **non-causal** and many off-the-shelf pretrained ASR models. These results raise a caution against simply claiming that a high-accuracy model is brain-like without evaluating its temporal dynamics. All plots and metrics produced by `evaluate.py` are saved in `results/`.

## Reproducing Paper Figures

Detailed analysis and figure generation notebooks are available in `notebooks/`:

- **Metrics:** See [notebooks/calculate_RMSE_MAE.ipynb](notebooks/calculate_RMSE_MAE.ipynb) to compute RMSE and MAE values comparing model trajectories to human data.

- **Figure 2 & 4:** See [notebooks/Fig2_4.ipynb](notebooks/Fig2_4.ipynb) for visualization of activation trajectories comparing human VWP fixations against model predictions. These figures show the characteristic temporal dynamics: early cohort competition followed by rhyme activation in causal models.

- **Figure 3:** See [notebooks/Fig3.ipynb](notebooks/Fig3.ipynb) for phoneme decoder analysis of internal layer representations across different model architectures.


## Citation

If you use this benchmark or code, please cite the paper. Temporary bibtex (to be updated upon acceptance):

```
@misc{htp2025,
  title={Do Machines Listen Like Humans? A Temporal Benchmark for Phonological Competition in End-to-End ASR},
  author={Anonymous},
  booktitle={Interspeech 2026},
  year={2026}
}
```

## License

This project is released under the MIT License. Human fixation data are used with permission from the original authors (Allopenna et al., 1998).

## Contact

For questions or issues, please open a GitHub issue or contact the authors (contact details in the paper).

---

**Acknowledgements**

This work builds on prior resources including the EARSHOT model and the Visual World Paradigm data (Allopenna et al., 1998). We thank the creators of those resources.
