"""Reliability diagram generation for a calibrated classifier."""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

ROOT = pathlib.Path(__file__).resolve().parents[2]


def plot_reliability(probs: np.ndarray, y: np.ndarray, out_path: pathlib.Path, title: str = "Reliability diagram") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888", linewidth=1)
    ax.plot(mean_pred, frac_pos, marker="o", color="#0f0f0f")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    import os
    tmp = out_path.with_name(out_path.name + ".tmp.png")
    fig.savefig(tmp, dpi=120, format="png")
    plt.close(fig)
    os.replace(tmp, out_path)
