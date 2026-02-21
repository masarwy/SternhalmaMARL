"""Plot helper for JSONL training metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_metrics(experiment_dir: Path, output_path: Path | None = None) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    metrics_path = experiment_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    rows = _load_jsonl(metrics_path)
    if not rows:
        raise RuntimeError(f"No rows found in {metrics_path}")

    episodes = [row.get("episode", idx + 1) for idx, row in enumerate(rows)]
    mean_returns = [row.get("episode_return_mean", 0.0) for row in rows]
    episode_steps = [row.get("episode_steps", 0) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(episodes, mean_returns, color="#0077b6", linewidth=2)
    axes[0].set_ylabel("Mean Return")
    axes[0].grid(alpha=0.3)

    axes[1].plot(episodes, episode_steps, color="#2a9d8f", linewidth=2)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Steps")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()

    out = output_path or (experiment_dir / "training_curves.png")
    fig.savefig(out, dpi=140)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics from JSONL logs.")
    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Directory containing metrics.jsonl",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    args = parser.parse_args()

    output = plot_metrics(args.experiment, args.output)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
