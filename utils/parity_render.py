#!/usr/bin/env python3
from __future__ import annotations

from io import BytesIO
from threading import Lock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


_RENDER_LOCK = Lock()


def render_binary_trajectory_png(states: np.ndarray) -> bytes:
    if states.ndim != 2 or not states.shape[0] or not states.shape[1]:
        raise ValueError("states must be a non-empty two-dimensional array")
    # pyplot owns process-global figure state and is not safe across cell threads.
    with _RENDER_LOCK:
        rounds, agents = states.shape
        fig_width = min(25.0, max(8.0, agents * 0.15 + 2.5))
        fig_height = min(20.0, max(6.0, rounds * 0.12 + 1.5))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        cmap = plt.cm.binary.copy()
        cmap.set_bad(color="#E0E0E0")
        aspect = (agents / rounds) * 0.8
        ax.imshow(
            states.astype(float),
            cmap=cmap,
            interpolation="nearest",
            origin="upper",
            aspect=aspect,
            vmin=0,
            vmax=1,
        )
        ax.set_xticks(range(0, agents, max(1, agents // min(12, agents))))
        ax.set_yticks(range(0, rounds, max(1, rounds // min(15, rounds))))
        ax.set_xlabel("Agentes", fontsize=10, fontweight="bold")
        ax.set_ylabel("Rodadas", fontsize=10, fontweight="bold")
        ax.set_title(
            "Simulação de Conformidade - Heatmap", fontsize=12, fontweight="bold"
        )
        ax.set_xticks(np.arange(-0.5, agents, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rounds, 1), minor=True)
        ax.grid(which="minor", color="#CCCCCC", linestyle="-", linewidth=0.5)
        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()
