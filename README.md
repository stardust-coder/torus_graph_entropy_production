# EEG Phase and Entropy Production

## Dataset and patient groups

This project assumes the **open EEG dataset from Chennu et al. (2016), *PLOS Computational Biology***, collected during propofol-induced sedation of healthy adults. The dataset contains resting-state EEG recordings for **20 participants** across four conditions: `baseline`, `mild`, `moderate`, and `recovery`. The two participant groups are defined by their behavioural responsiveness during sedation (not by the recording state):

- **drowsy (n = 7):** `3 6 7 8 10 14 26`
- **responsive (n = 13):** `2 5 9 13 18 20 22 23 24 25 27 28 29`

Use these exact, mutually exclusive patient IDs consistently in all commands. The phase arrays passed to this program must be prepared from the public EEG recordings by the user-provided `data.py:get_phase_data(...)` loader; this README does not imply the raw EEG is bundled with the code.

**Reference:** Chennu, S., O’Connor, S., Adapa, R., Menon, D. K., & Bekinschtein, T. A. (2016). *Brain Connectivity Dissociates Responsiveness from Drug Exposure during Propofol-Induced Transitions of Consciousness*. PLOS Computational Biology, 12(1), e1004669. https://doi.org/10.1371/journal.pcbi.1004669

**Open EEG dataset:** University of Cambridge Data Repository, *Research data supporting “Brain connectivity during propofol sedation”*. https://doi.org/10.17863/CAM.68959 (https://www.repository.cam.ac.uk/handle/1810/252736)

There are two patient groups: **drowsy** and **responsive**. Every ID is an integer and must appear in exactly one group. For each patient, each band and each state (`baseline`, `mild`, `moderate`, `recovery`), the program calls `data.py:get_phase_data(patient_id, band, state, value_type)` returning a `(N,d)` NumPy array of phases in radians.

Run:

```bash
python -u main.py \
  --experiment batch-ep \
  --band alpha \
  --drowsy-ids 3 6 7 8 10 14 26 \
  --responsive-ids 2 5 9 13 18 20 22 23 24 25 27 28 29 \
  --states baseline mild moderate recovery \
  --models ktg \
  --value-type absolute \
  --group-lasso \
  --lambda-min-ratio 0.00001 \
  --n-lambdas 120 \
  --max-iter 10000 \
  --n-jobs 64 \
  --tol 1e-5 \
  --lambda-solver parallel
```

The program fits each cell separately, uses AIC computed after unpenalized support refitting for lambda selection, and computes EP from the selected unpenalized support-refit coefficients. Without `--group-lasso`, unpenalized MLE is used. Data loading stays in `data.py`; replace its `get_phase_data` implementation. Example with an explicit lambda: `--lambda 0.01`.

CSV includes `group`, `patient_id`, `band`, `state`, `model`, `ep_nats_per_step` and estimation diagnostics. Plots are saved separately **for each model and band** under `results/plots/`: group × state means, group × state boxplots with individual measurements, and individual-patient trajectories.

For formal two-way inference, factors are **group (drowsy/responsive)** and **state (four levels)**. Analyze each band and each model separately. As states are typically repeated measures within patient, a Gaussian mixed-effects model with a patient random intercept is more appropriate than an independence-assuming two-way ANOVA. For comparing fixed effects using mixed-model LRTs, fit models using ML rather than REML. Statistical tests are not implemented in this release; plots are descriptive.

Missing patient/state combinations require deliberate handling. Group labels are patient-level; they are not the same factor as states.

## Run each band independently

`--experiment batch-ep` requires exactly one `--band`. Results are not pooled across bands.

The default output location is `results/<band>/ep.csv`, with figures in

`results/<band>/plots/`. To analyze multiple bands, run a separate command

for each band, for example:

```bash
for band in delta theta alpha beta gamma; do
  python main.py --experiment batch-ep --band "$band" \
  --drowsy-ids 3 6 7 8 10 14 26 --responsive-ids 2 5 9 13 18 20 22 23 24 25 27 28 29 \
  --models ktg artg --group-lasso
done
```

Every fit, AIC lambda search, EP summary, CSV, and plot is independently

computed for the selected band. The two-way factors within each band are

patient group and state; KTG and AR-TG are separate.

## Phase value mode and conjugate time reversal

Specify `--value-type absolute` (default) or `--value-type relative` to pass

the selected phase representation into the `data.py` loader as its fourth argument:

```python
def get_phase_data(patient_id: int, band: str, state: str, value_type: str):
    # value_type is "absolute" or "relative"
    ...
```

Use `--conjugate` to evaluate the reverse log likelihood using

`-X[::-1]` instead of `X[::-1]`. Forward likelihood and model fitting

continue to use the original input. Negation is circular phase conjugation

(`exp(1j * phi).conjugate() = exp(-1j * phi)`). This changes the

notion of time reversal underlying EP; it is not the usual reversal-only

EP. Neither mode drops the von Mises normalizers.

Example:

```bash
python main.py --experiment batch-ep --band alpha \
  --drowsy-ids 3 6 7 8 10 14 26 --responsive-ids 2 5 9 13 18 20 22 23 24 25 27 28 29 \
  --value-type relative --conjugate --group-lasso
```

CSV columns `value_type` and `conjugate` document the setting.

Run each phase-value/conjugation setting to a distinct output CSV to avoid overwrites.

## Progress logging

`batch-ep` prints progress to stderr by default. It includes the current

patient/state/model, AIC lambda search counts, elapsed time, and final EP.

`--lambda-progress-every 1` reports every lambda candidate; `--quiet` suppresses

progress messages. To ensure unbuffered standard output, use `python -u main.py ...`.

```bash
python -u main.py --experiment batch-ep --band alpha \
  --drowsy-ids 3 6 7 8 10 14 26 --responsive-ids 2 5 9 13 18 20 22 23 24 25 27 28 29 \
  --group-lasso --lambda-progress-every 5
```

## Parallel AIC search (without warm starts)

`--outer-jobs` parallelizes patient/state/model cells using threads, and

`--lambda-jobs` parallelizes independent lambda fits + support refits using

joblib's loky processes. Each lambda starts from zero (no warm start).

For a 128-logical-core machine, start with:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u main.py \
  --experiment batch-ep --band alpha \
  --drowsy-ids 3 6 7 8 10 14 26 --responsive-ids 2 5 9 13 18 20 22 23 24 25 27 28 29 \
  --models ktg artg --group-lasso --outer-jobs 4 --lambda-jobs 16
```

Increase parallelism after measuring wall time and memory. Logs for AIC candidates

are printed after a parallel batch is completed (not as each worker finishes).

The data loader must be safe for concurrent thread calls. Avoid multithreaded BLAS

oversubscription.

## Single-pool parallel batch EP (current version)

The `batch-ep` experiment now uses **one dynamically scheduled process pool**.

Every `(patient_id, state, model, lambda)` candidate is an independent job,

starting from zero (no warm starts). Each worker computes the Group Lasso

estimate and the support-refitted AIC. The parent selects the minimum AIC

for each `(patient_id, state, model)` and computes EP with **the Group Lasso

estimate**, not the refit estimate. The data loader is called once per

`(patient_id, band, state)`; arrays are saved temporarily and shared read-only

via memory-mapped `.npy` files. Results remain separated by band.

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u main.py --experiment batch-ep --band alpha \
  --drowsy-ids 3 6 7 8 10 14 26 --responsive-ids 2 5 9 13 18 20 22 23 24 25 27 28 29 \
  --models ktg artg --states baseline mild moderate recovery \
  --group-lasso --n-jobs 128 --value-type relative --conjugate
```

`--n-jobs` replaces the previous `--outer-jobs` and `--lambda-jobs` flags.

Set it to your **allocated** cores, not necessarily the server-wide count.

A starting benchmark at 32 or 64 workers is advisable because 128 Python

processes require considerable RAM. Candidate-completion messages appear

in completion order rather than lambda order.

### AIC selection audit and diagnostics

In `batch-ep`, positive Group Lasso lambda values are searched by default.

Unlike the previous version, unregularized `lambda=0` is **not** added

implicitly to a `--group-lasso` AIC search. Set `--include-zero-lambda`

to include the unregularized model intentionally. Note that excluding zero

changes the model-selection candidate set; it does not guarantee sparse output.

Candidates are considered eligible only when both ISTA/MLE and the

support-restricted refit converged and AIC is finite. If no candidate is

eligible, the batch exits with an error rather than silently reporting an

unconverged optimum. Increase `--max-iter` or adjust convergence tolerance.

`batch-ep` automatically writes `aic_path.csv` beside `ep.csv`, with one

row per lambda candidate, including support size and convergence flags. Use

`--aic-path-csv path/to/file.csv` to override its location.

Example:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u main.py --experiment batch-ep --band alpha \
  --drowsy-ids 3 6 7 8 10 14 26 --responsive-ids 2 5 9 13 18 20 22 23 24 25 27 28 29 --models ktg artg \
  --group-lasso --n-jobs 64 --max-iter 5000
```

**Statistical caution:** AIC with post-selection refit may still favor

nearly or fully dense models, particularly with strongly predictable data.

This is not automatically a software error; inspect the entire AIC path.

Selection of supports on the same data and conditional in-sample EP can

introduce optimism. Consider held-out validation for scientific comparisons.

### AIC path plots

For `--experiment batch-ep --group-lasso`, the program saves an additional

PNG per (group, patient, band, state, model) into the standard `plots/`

directory. The plot shows refit AIC versus lambda, marks the minimum AIC

among converged candidates, and displays failed candidates separately.

It reads the automatically generated `aic_path.csv`. `--no-plots` disables

these plots together with the existing EP plots. A positive lambda grid is

shown on a logarithmic-style (`symlog`) axis, which also handles lambda=0.

### AIC path CSV: streaming writes

In `batch-ep`, the candidate diagnostic CSV (`--aic-path-csv`, default

`results/<band>/aic_path.csv`) is created before the worker pool runs.

Each finished lambda candidate appends its diagnostic row immediately (with a

flush), so an incomplete run still leaves a partial CSV. Candidate rows are

ordered by task completion, not by lambda; sort by `lambda_` when plotting.

The final plot files are generated after the full batch has completed.

### AIC diagnostics (updated)

`--max-iter` now defaults to **10000** (previously 2000).

The per-patient AIC PNG uses **selected edge count** (`active_groups`) on the horizontal axis, with each lambda candidate plotted as a separate point. The red star marks the minimum AIC among converged candidates; crosses indicate unconverged candidates. Output filenames end in `_edges_aic.png`.

With `lags>1`, `active_groups` counts lag-specific directed edges (one group for each lag/source/target). Same number of active groups does not imply identical supports.

## 高速化版の実装（重要）

- Group Lasso の最適化は monotone/restarted **FISTA** とバックトラッキング。

- ラインサーチの候補値では損失だけを計算し、不要な勾配計算を省略。

- 各ワーカーでデータ・正弦余弦特徴量を LRU キャッシュ（ワーカー間では共有しない）。

- `batch-ep` は2段階処理：各 λ で辺集合を抽出 → **収束済みの同じ辺集合を一度だけ**非正則化最尤Refit。AIC = −2 × 総対数尤度 + 2 × 自由係数数。

- 最小AICの **Refit係数** で対数尤度とEPを計算（Group Lasso係数は最終EPに不使用）。

- PNGは辺数–AICの全体図と、0辺の外れ値を除いた拡大図 `_edges_aic_zoom.png` を保存。

- `aic_path.csv` はRefit段階のあとで埋まります（最初にヘッダーだけ作成されます）。長時間の計算中にAIC行がすぐには増えません。

- 128同時実行は非常に大きなメモリを必要とします。まず `--n-jobs 4` から `16` で比較してください。`--lambda-min-ratio 1e-5` は探索を密グラフ側へ広げるオプションですが、収束を保証しません。

- 10 epoch を単純連結すると境界9箇所の偽の遷移も尤度に含まれるため、将来の正式な実験では各epoch境界を除外してください。

### λ solver modes: independent parallel versus sequential warm-start paths

`batch-ep` supports `--lambda-solver {parallel,path}` (default: `parallel`).

- `--lambda-solver parallel`: each `(patient,state,band,model,λ)` is an independent process-pool job. All candidate fits start from zero. Up to `--n-jobs` lambda jobs can run concurrently.

- `--lambda-solver path`: each `(patient,state,band,model)` is a single job; its λ grid is solved from **largest to smallest λ**, initializing each FISTA fit with the preceding fit's coefficients. Up to `min(--n-jobs, number_of_cells)` paths run concurrently. For two patients, baseline, KTG only, this means **at most two support-stage workers** even with `--n-jobs 128`. The refit stage still runs individual unique-support jobs in parallel using `--n-jobs`.

Both modes use the same λ grid, support threshold, deduplication by complete support mask, unpenalized restricted-MLE refit, AIC criterion, and refitted coefficients for final EP. Results can differ slightly due to optimization tolerances and finite-iteration stopping; compare the `ista_converged` and `active_groups` columns in their `aic_path.csv` outputs, as well as elapsed time. Each run overwrites its specified CSVs; use **different `--output-csv` paths** for side-by-side comparisons. Progress from `path` mode is reported when a path job returns (not after each individual λ within that path).

Examples:

```bash
python -u main.py --experiment batch-ep --band alpha \
  --drowsy-ids 3 6 7 8 10 14 26 --responsive-ids 2 5 9 13 18 20 22 23 24 25 27 28 29 --states baseline --models ktg \
  --group-lasso --lambda-min-ratio 1e-5 --n-lambdas 120 \
  --lambda-solver parallel --n-jobs 8 --output-csv results/alpha/parallel_ep.csv
python -u main.py --experiment batch-ep --band alpha \
  --drowsy-ids 3 6 7 8 10 14 26 --responsive-ids 2 5 9 13 18 20 22 23 24 25 27 28 29 --states baseline --models ktg \
  --group-lasso --lambda-min-ratio 1e-5 --n-lambdas 120 \
  --lambda-solver path --n-jobs 8 --output-csv results/alpha/path_ep.csv
```

### Progress while using `--lambda-solver path`

Each worker prints `[path cell=... k/K] elapsed=... lambda=... edges=... converged=... iterations=...` every `--lambda-progress-every` candidate solves (default 5), and at the end of its path. Because joblib sends the **whole** path back only after completion, the parent `[supports x/y]` counter still updates at the end of each complete path. The new worker-side messages allow monitoring progress while the sequential path is running. `--no-progress` suppresses these messages where supported. These prints do not change computation or output ordering.

## Fast matrix products / optimality diagnostics

The group-Lasso objective remains **mean negative conditional log-likelihood + λ × group norm**. The fit now uses 2D BLAS matrix products in place of multi-index `einsum`, keeping the same likelihood and gradients. The FISTA convergence flag is based on the relative **proximal gradient mapping residual** rather than on consecutive coefficient changes. It is evaluated periodically (every 25 iterations and whenever the coefficient change is small), and once at termination for nonconverged fits.

The `batch-ep` AIC diagnostics CSV additionally contains `fit_seconds`, `backtracking_trials`, and `kkt_residual` (relative proximal gradient mapping norm). Path-mode progress logs display these diagnostics. `--tol` now specifies the proximal-gradient residual tolerance for Group Lasso. As before, AIC is based on a **separate, unpenalized restricted MLE refit** for each unique support, and EP uses the chosen refit coefficients. Both `--lambda-solver parallel` and `--lambda-solver path` remain supported.

Important: no automatic stopping of the λ path based on observing an apparent AIC minimum has been introduced, since AIC need not be unimodal and refit occurs only after the support stage. For an initial exploratory run, use fewer λ candidates or a less extreme `--lambda-min-ratio`, and expand if needed. For 25,000 time points, limit BLAS thread counts (`OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1`) to avoid oversubscription. A genuinely optimal wall-clock performance improvement on 25,000-sample EEG data has **not** yet been benchmarked.

### Dual EP output (batch-ep)

Each fit now evaluates both `ep_normal` (time reversal only) and `ep_conjugate`

(time reversal plus phase sign reversal) **without rerunning estimation**.

The legacy `ep_nats_per_step` column remains for compatibility; it mirrors

`ep_normal` by default or `ep_conjugate` when `--conjugate` is passed.

The `--conjugate` flag therefore changes only the legacy column and does not

affect either of the new EP values. Plots are saved separately with

`_normal_` and `_conjugate_` in their filenames. Nothing is saved as a model.
