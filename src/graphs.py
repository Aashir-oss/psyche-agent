"""Matplotlib charts. matplotlib is imported lazily so `import src.graphs` is light."""
from pathlib import Path

from .config import GRAPHS_DIR


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            f"Graphs need matplotlib: pip install -r requirements.txt ({e.name})."
        )
    return plt


def severity_trend(dates: list[str], scores: list[int], out_name: str = "severity.png") -> str:
    """Line chart of severity 1-10 across sessions. Returns the PNG path."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(dates, scores, marker="o", linewidth=2)
    ax.set_ylim(0.5, 10.5)
    ax.set_ylabel("Severity (1-10)")
    ax.set_xlabel("Session date")
    ax.set_title("How intense things felt, over time")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    out = GRAPHS_DIR / out_name
    try:
        fig.savefig(out, dpi=120)
    finally:
        plt.close(fig)
    return str(out)


def perspective_agreement(items: list[tuple[str, int]], out_name: str = "agreement.png") -> str:
    """Bar chart of each perspective's confidence %. Returns the PNG path."""
    plt = _plt()
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(names, vals)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Confidence %")
    ax.set_title("How confident each perspective was")
    for bar, v in zip(bars, vals):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v}%", va="center")
    fig.tight_layout()
    out = GRAPHS_DIR / out_name
    try:
        fig.savefig(out, dpi=120)
    finally:
        plt.close(fig)
    return str(out)


def scale_trend(dates: list[str], scores: list[int], scale_name: str,
               bands: list[tuple], out_name: str = "scale.png") -> str:
    """Line chart of GAD-7/PHQ-9 scores with shaded severity bands.

    bands: list of (lo, hi, label, color). Returns the PNG path.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for lo, hi, label, color in bands:
        ax.axhspan(lo, hi + 1, facecolor=color, alpha=0.55, label=label)
    ax.plot(dates, scores, marker="o", linewidth=2, color="#1a1a2e", zorder=5)
    max_hi = max([hi for _, hi, _, _ in bands] or [27])
    ax.set_ylim(-0.5, max_hi + 1.5)
    ax.set_ylabel(f"{scale_name} score")
    ax.set_xlabel("Check-in date")
    ax.set_title(f"{scale_name} scores over time")
    if bands:
        ax.legend(loc="upper left", fontsize=8, ncol=len(bands))
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    out = GRAPHS_DIR / out_name
    try:
        fig.savefig(out, dpi=120)
    finally:
        plt.close(fig)
    return str(out)


def parse_confidence(text: str) -> int:
    """Extract the first 0-100 number from a CONFIDENCE string; default 50."""
    import re
    m = re.search(r"(\d{1,3})", text or "")
    if not m:
        return 50
    return max(0, min(100, int(m.group(1))))
