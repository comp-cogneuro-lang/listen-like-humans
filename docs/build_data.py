#!/usr/bin/env python3
"""
Build the JSON payloads that drive the interactive GitHub Pages site.

This reads the per-model ``competition_mean.csv`` trajectory files plus the
human VWP reference (``notebooks/INPUT/amt_human_mean.csv``) and emits three
small JSON files into ``docs/data/``:

    trajectories.json  - per-model activation curves (+ human reference) for plotting
    human.json         - the human reference curves on their own
    metrics.json       - RMSE / MAE / correlation per model per competitor type

The RMSE/MAE pipeline is a faithful port of notebooks/calculate_RMSE_MAE.ipynb
so the published numbers match the paper's notebook (same model->csv mapping,
same tmult time conversion, same Cross generation, same alignment by min length).

Run from the repo root:

    python docs/build_data.py
"""

import json
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_OUT = os.path.join(HERE, "data")
HUMAN_CSV = os.path.join(PROJECT_ROOT, "notebooks/INPUT/amt_human_mean.csv")

# --------------------------------------------------------------------------- #
# Model registry
#
# tmult converts the CSV "Time" column to milliseconds:
#   - trained-from-scratch models store a frame index (0..100) -> x10 ms
#   - foundation models already store milliseconds                -> x1
# --------------------------------------------------------------------------- #
MODELS = [
    # key,            display name,           type,         group,       tmult, csv
    ("baseline",        "Causal-LSTM",          "causal",     "trained",    10,
     "experiments/en_words_ku_baseline/Baseline/training/competition_mean.csv",
     "Single-layer unidirectional LSTM"),
    ("causal-2lstm",    "Causal-2LSTM",         "causal",     "trained",    10,
     "experiments/en_words_ku_causal_2lstm/causal-2LSTM/training/competition_mean.csv",
     "Two-layer unidirectional LSTM"),
    ("causal-cnn",      "Causal-CNN",           "causal",     "trained",    10,
     "experiments/en_words_ku_causal_cnn/causal-cnn/training/competition_mean.csv",
     "1D convolutions with causal padding"),
    ("causal-rcnn",     "Causal-RCNN",          "causal",     "trained",    10,
     "experiments/en_words_ku_causal_rcnn/causal-rcnn/training/competition_mean.csv",
     "1D CNN + unidirectional LSTM"),
    ("causal-trans",    "Causal-Transformer",   "causal",     "trained",    10,
     "experiments/en_words_ku_causal_trans/causal-trans/training/competition_mean.csv",
     "Transformer with causal self-attention"),
    ("causal-ctrans",   "Causal-ConvTransformer", "causal",   "trained",    10,
     "experiments/en_words_ku_causal_convtrans/caconvtrans2/training/competition_mean.csv",
     "Conformer-style with causal context"),

    ("noncausal-2lstm", "NonCausal-BiLSTM",     "noncausal",  "trained",    10,
     "experiments/en_words_ku_2lstmbi/noncausal-2LSTM/training/competition_mean.csv",
     "Bidirectional LSTM (full context)"),
    ("noncausal-rcnn",  "NonCausal-RCNN*",      "noncausal",  "trained",    10,
     "experiments/en_words_ku_rcnn/noncausal-rcnn/training/competition_mean.csv",
     "Causal LSTM on a non-causal centered CNN (25-frame receptive field, ±12-frame look-ahead)"),
    ("noncausal-cnn",   "NonCausal-CNN",        "noncausal",  "trained",    10,
     "experiments/en_words_ku_cnn/noncausal-cnn/training/competition_mean.csv",
     "Standard 1D CNN (full context)"),
    ("noncausal-trans", "NonCausal-Transformer", "noncausal", "trained",    10,
     "experiments/en_words_ku_trans/noncausal-trans/training/competition_mean.csv",
     "Full bidirectional self-attention"),
    ("noncausal-ctrans", "NonCausal-ConvTransformer", "noncausal", "trained", 10,
     "experiments/en_words_ku_convtrans/noncausal-convtrans/training/competition_mean.csv",
     "Conformer-style with full context"),

    ("wav2vec2",        "Foundation-wav2vec2",  "foundation", "foundation",  1,
     "experiments/wav2vec2/competition_mean.csv",
     "Pretrained wav2vec 2.0 (CTC), no fine-tuning"),
    ("hubert",          "Foundation-HuBERT",    "foundation", "foundation",  1,
     "experiments/hubert/facebook/hubert-large-ls960-ft/competition_mean.csv",
     "Pretrained HuBERT-large (CTC), no fine-tuning"),
    ("whisper",         "Foundation-Whisper",   "foundation", "foundation",  1,
     "experiments/whisper/competition_mean.csv",
     "Pretrained Whisper (attention token probs), no fine-tuning"),
]

ITEM_TYPES = ["Target", "Cohort", "Rhyme", "Unrelated", "Cross"]
TMAX = 1000  # ms


# --------------------------------------------------------------------------- #
# Pipeline ported from notebooks/calculate_RMSE_MAE.ipynb
# --------------------------------------------------------------------------- #
def interpolate_missing_time_steps(df):
    time_column = df.columns[0]
    prob_cols = df.columns[1:]
    df[time_column] = df[time_column].astype(int)
    full_range = range(df[time_column].min(), df[time_column].max() + 1)
    out = df.set_index(time_column).reindex(full_range).reset_index()
    out[prob_cols] = out[prob_cols].interpolate()
    return out


def read_csv_for_metrics(file_path, tmult=1, generate_cross=True, tmax=TMAX):
    """Mirror of the notebook's read_csv with the exact flags used there
    (normalize=False, rescale=False, apply_lcr=False)."""
    df = pd.read_csv(file_path)

    if "Cross" not in df.columns:
        sum_columns = [c for c in ["Target", "Cohort", "Rhyme", "Unrelated"] if c in df.columns]
        if generate_cross:
            total_sum = df[sum_columns].sum(axis=1)
            sum_max = total_sum.max()
            df["Cross"] = sum_max - (total_sum / sum_max)
            df["Cross"] = df["Cross"].clip(lower=0)
            zero_mask = df["Cross"].eq(0)
            df["Cross"] = df["Cross"].mask(zero_mask.cumsum().gt(0), 0)
        else:
            df["Cross"] = np.nan

    df["Time"] = df["Time"] * tmult
    df = interpolate_missing_time_steps(df)
    df = df[df["Time"] <= tmax]
    return df


def calculate_metrics(human_df, comp_df):
    min_len = min(len(human_df), len(comp_df))
    h = human_df.iloc[:min_len].reset_index(drop=True)
    c = comp_df.iloc[:min_len].reset_index(drop=True)

    metrics = {}
    pooled_true, pooled_pred = [], []
    for item in ITEM_TYPES:
        if item in h.columns and item in c.columns:
            y_true = h[item].astype(float)
            y_pred = c[item].astype(float)
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            if y_true.std() == 0 or y_pred.std() == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(y_true, y_pred)[0, 1])
            metrics[item] = {"RMSE": round(rmse, 4), "MAE": round(mae, 4),
                             "Corr": round(corr, 4)}
            pooled_true.append(y_true.values)
            pooled_pred.append(y_pred.values)

    if pooled_true:
        yt = np.concatenate(pooled_true)
        yp = np.concatenate(pooled_pred)
        metrics["Overall"] = {
            "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "MAE": round(float(np.mean(np.abs(yt - yp))), 4),
            "Corr": round(float(np.corrcoef(yt, yp)[0, 1]), 4),
        }
    return metrics


# --------------------------------------------------------------------------- #
# Trajectory payload (native resolution, for plotting)
# --------------------------------------------------------------------------- #
def load_trajectory(csv_path, tmult):
    df = pd.read_csv(csv_path)
    time_ms = (df["Time"].astype(float) * tmult).round(1).tolist()
    series = {}
    for col in df.columns[1:]:
        series[col] = df[col].astype(float).round(4).where(df[col].notna(), None).tolist()
    return time_ms, series


def main():
    os.makedirs(DATA_OUT, exist_ok=True)

    # Human reference -----------------------------------------------------
    human_raw = pd.read_csv(HUMAN_CSV)
    human_time = human_raw["Time"].astype(float).round(1).tolist()
    human_series = {c: human_raw[c].astype(float).round(4).tolist()
                    for c in human_raw.columns[1:]}
    human_payload = {"time_ms": human_time, "series": human_series}

    human_metrics_df = read_csv_for_metrics(HUMAN_CSV, tmult=1)

    # Models --------------------------------------------------------------
    trajectories = {}
    metrics = {}
    skipped = []
    for key, name, mtype, group, tmult, rel, desc in MODELS:
        path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.exists(path):
            skipped.append((key, rel))
            continue

        time_ms, series = load_trajectory(path, tmult)
        trajectories[key] = {
            "name": name, "type": mtype, "group": group,
            "description": desc, "time_ms": time_ms, "series": series,
        }

        comp_df = read_csv_for_metrics(path, tmult=tmult)
        metrics[key] = {
            "name": name, "type": mtype, "group": group,
            "metrics": calculate_metrics(human_metrics_df, comp_df),
        }
        print(f"  ok  {key:18s} <- {rel}")

    for key, rel in skipped:
        print(f"  SKIP {key:17s} (missing {rel})")

    # Write ---------------------------------------------------------------
    trajectories_payload = {"human": human_payload, "models": trajectories}
    metrics_payload = {"items": ITEM_TYPES + ["Overall"], "models": metrics}

    with open(os.path.join(DATA_OUT, "human.json"), "w") as f:
        json.dump(human_payload, f, separators=(",", ":"))
    with open(os.path.join(DATA_OUT, "trajectories.json"), "w") as f:
        json.dump(trajectories_payload, f, separators=(",", ":"))
    with open(os.path.join(DATA_OUT, "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, separators=(",", ":"))

    # Also emit a plain <script>-loadable bundle so the page never needs fetch()
    # (works over file:// and inside sandboxed webview previews).
    bundle = (
        "// Auto-generated by docs/build_data.py — do not edit by hand.\n"
        "window.HTP_DATA = {\n"
        "  trajectories: " + json.dumps(trajectories_payload, separators=(",", ":")) + ",\n"
        "  metrics: " + json.dumps(metrics_payload, separators=(",", ":")) + "\n"
        "};\n"
    )
    with open(os.path.join(HERE, "data.js"), "w") as f:
        f.write(bundle)

    print(f"\nWrote {len(trajectories)} models to {DATA_OUT} and docs/data.js")


if __name__ == "__main__":
    main()
