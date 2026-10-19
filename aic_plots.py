"""Save per-patient, per-state, per-model number of selected edges vs refit-AIC diagnostics."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


def save_aic_plots(aic_csv, output_dir):
    """Plot AIC paths; return paths saved. Unconverged candidates are not eligible.

    Parameters
    ----------
    aic_csv : path
        The per-candidate diagnostic CSV produced by batch_ep.run_batch_ep.
    output_dir : path
        Destination directory; one PNG per group/patient/band/state/model.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with Path(aic_csv).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    keys = ('group', 'patient_id', 'band', 'state', 'model')
    cells = defaultdict(list)
    for row in rows:
        cells[tuple(row[k] for k in keys)].append(row)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for key, entries in cells.items():
        points = []
        for row in entries:
            lam = float(row['lambda_'])
            aic = float(row['aic'])
            ok = (str(row['ista_converged']).lower() == 'true'
                  and str(row['refit_converged']).lower() == 'true'
                  and math.isfinite(aic))
            points.append((int(row['active_groups']), aic, ok, lam))
        points.sort(key=lambda p: (p[0], p[3]))
        valid = [(x, y, lam) for x, y, ok, lam in points if ok]
        invalid = [(x, y) for x, y, ok, lam in points if not ok and math.isfinite(y)]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if valid:
            ax.scatter([p[0] for p in valid], [p[1] for p in valid],
                       s=26, alpha=.8, label='Converged / eligible')
            best_edges, best_aic, best_lambda = min(valid, key=lambda p: (p[1], p[2]))
            ax.scatter([best_edges], [best_aic], color='tab:red', marker='*', s=160,
                       zorder=4, label=f'Min AIC: {best_edges} edges (λ={best_lambda:.4g})')
        if invalid:
            ax.scatter([p[0] for p in invalid], [p[1] for p in invalid],
                       color='tab:gray', marker='x', label='Not converged / excluded')
        if not valid:
            ax.text(.5, .5, 'No converged candidate', transform=ax.transAxes,
                    ha='center', va='center')
        ax.set_xlabel('Number of selected directed edges (per lag)')
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_ylabel('Refit AIC (lower is better)')
        group, patient, band, state, model = key
        ax.set_title(f'{model.upper()} | {band} | {state} | patient {patient} ({group})')
        ax.grid(alpha=.25)
        if valid or invalid:
            ax.legend(fontsize=8)
        fig.tight_layout()
        name = '_'.join(key).replace('/', '-') + '_edges_aic.png'
        path = output_dir / name
        fig.savefig(path, dpi=160)
        saved.append(path)
        # A zero-edge model can be millions of AIC units away, hiding the
        # structure-selection region. Save a second zoom without that outlier.
        zoom_points = [p for p in points if p[0] > 0 and math.isfinite(p[1])]
        if len(zoom_points) >= 2 and len({p[0] for p in zoom_points}) >= 2:
            nonzero_aics = [p[1] for p in zoom_points]
            low, high = min(nonzero_aics), max(nonzero_aics)
            padding = max(1.0, (high - low) * .05)
            ax.set_ylim(low - padding, high + padding)
            ax.set_xlim(min(p[0] for p in zoom_points) - 1,
                        max(p[0] for p in zoom_points) + 1)
            zoom_path = output_dir / name.replace('_edges_aic.png', '_edges_aic_zoom.png')
            fig.savefig(zoom_path, dpi=160)
            saved.append(zoom_path)
        plt.close(fig)
    return saved
