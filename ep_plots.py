"""Group × state descriptive EP plots, separated by model and band."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

STATE_ORDER = ('baseline', 'mild', 'moderate', 'recovery')
GROUP_ORDER = ('drowsy', 'responsive')


def save_ep_plots(csv_path, output_dir, *, models=('ktg', 'artg')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with Path(csv_path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('EP CSV is empty')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    bands = list(dict.fromkeys(r['band'] for r in rows))
    states = [s for s in STATE_ORDER if any(r['state'] == s for r in rows)]
    groups = [g for g in GROUP_ORDER if any(r['group'] == g for r in rows)]
    for model in models:
        for band in bands:
            subset = [r for r in rows if r['model'] == model and r['band'] == band]
            if not subset:
                continue

            for ep_col, ep_label in (('ep_normal', 'normal'), ('ep_conjugate', 'conjugate')):
                paths.extend(_save_ep_variant(subset, output_dir, model, band, states, groups, ep_col, ep_label, plt))
    return paths


def _save_ep_variant(subset, output_dir, model, band, states, groups, ep_col, ep_label, plt):
    paths = []
    def values(group, state):
        return np.array([float(row[ep_col]) for row in subset
                         if row['group'] == group and row['state'] == state
                         and np.isfinite(float(row[ep_col]))])
    stem = f'{model}_{band}_{ep_label}'
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(states))
    for group in groups:
        means = [np.mean(v) if (v := values(group, state)).size else np.nan for state in states]
        ax.plot(positions, means, marker='o', linewidth=2, label=group)
    ax.set_xticks(positions, states)
    ax.set_ylabel('Mean EP (nats / step)')
    ax.set_title(f'{model.upper()} / {band}: {ep_label} EP, group × state')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = output_dir / f'{stem}_group_state_means.png'
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.32
    for ig, group in enumerate(groups):
        pos, vals = [], []
        for si, state in enumerate(states):
            v = values(group, state)
            if len(v):
                pos.append(si + 1 + (ig - (len(groups)-1)/2)*width)
                vals.append(v)
        if vals:
            ax.boxplot(vals, positions=pos, widths=width*0.78, patch_artist=False,
                       medianprops={'linewidth': 2})
            for x, v in zip(pos, vals):
                ax.scatter(np.full(len(v), x), v, alpha=.65, s=15)
    ax.set_xticks(np.arange(1, len(states)+1), states)
    ax.set_ylabel('EP (nats / step)')
    ax.set_title(f'{model.upper()} / {band}: {ep_label} EP by group and state')
    ax.text(.02, .98, 'left: drowsy    right: responsive', transform=ax.transAxes,
            va='top', fontsize=9)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    path = output_dir / f'{stem}_group_state_boxplots.png'
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, axs = plt.subplots(1, len(groups), figsize=(5*len(groups), 4.5), squeeze=False,
                            sharey=True)
    for ax, group in zip(axs[0], groups):
        ids = list(dict.fromkeys(r['patient_id'] for r in subset if r['group'] == group))
        for pid in ids:
            ys = [next((float(r[ep_col]) for r in subset if
                        r['group'] == group and r['patient_id'] == pid and r['state'] == state),
                       np.nan) for state in states]
            ax.plot(range(len(states)), ys, marker='o', linewidth=1, alpha=.5)
        ax.set_title(group)
        ax.set_xticks(range(len(states)), states, rotation=20)
        ax.grid(axis='y', alpha=.25)
    axs[0,0].set_ylabel('EP (nats / step)')
    fig.suptitle(f'{model.upper()} / {band}: {ep_label} EP within-patient trajectories')
    fig.tight_layout()
    path = output_dir / f'{stem}_patient_trajectories.png'
    fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    return paths
