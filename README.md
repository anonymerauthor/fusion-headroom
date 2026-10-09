# Information, Not Diversity: Decomposing Fusion Gain in Classifier Committees

Code and results for the paper. Under the logarithmic loss, the largest gain of any fusion rule over the best member of
a committee (the *fusion headroom*) equals the recalibration gain of that member plus the conditional mutual information
between the label and the other members given the best one. A given rule realises the headroom minus an approximation
gap and an estimation gap. This repository estimates these terms for tabular committees on OpenML-CC18 and for
committees of deep networks on CIFAR-10/100, and reproduces all tables and figures of the paper.

## Installation

Python 3.11. The tabular experiments run on CPU; training the deep pools needs a GPU.

```bash
pip install -r requirements.txt
```

## Repository layout

```
src/fh/
  info.py         variational bounds on conditional entropies, headroom decomposition, learning-curve correction,
                  Shapley information values
  accounting.py   gain accounting (headroom = gain + approximation gap + estimation gap)
  pruning.py      pruning criteria (information forward selection, CMIM filter, L1 stacking, greedy ES, ...)
  diversity.py    the ten classic diversity measures and three label-free measures
  synth.py        Gaussian committees with closed-form ground truth
  rules.py        fusion rules (means, voting, greedy ensemble selection, stacking, KNORA-E, fitted pools, ...)
  committee.py    the base learners and out-of-fold cross-fitting used to build the tabular committees
  data.py         access to the committee files and the fixed development/test split of the data sets
src/deep/models.py  CIFAR architectures
scripts/          experiment, table and figure scripts (see below)
results/          per-instance results of all experiments (CSV), from which every table and figure is built
data/rule_results/  realised test losses of the thirteen fusion rules per data set and fold
outputs/          generated LaTeX tables and PDF figures
```

## Reproducing the tables and figures from the stored results

All tables and figures can be rebuilt in a few minutes on a CPU from `results/` and `data/rule_results/`:

```bash
python scripts/tab_synth.py          # Table I    estimator validation on Gaussian committees
python scripts/rq3_v2.py             # Table II   prediction of realised gains (+ supplementary baselines table)
python scripts/tab_accounting.py     # Table III  gain accounting
python scripts/tab_pruning_v3.py     # Table IV   pruning (+ supplementary pruning curves)
python scripts/tab_deep.py           # Tables V-VI deep committees
python scripts/fig_toys.py           # Fig. 1     diversity versus headroom on controlled committees
python scripts/fig_rq3.py            # Fig. 2     information terms versus diversity
python scripts/fig_accounting.py     # Fig. 3     where the headroom goes
python scripts/supp_tables.py        # supplementary: learning-curve estimates, deep pruning
python scripts/supp_accuracy.py      # supplementary: accuracy gains
python scripts/tab_pruning_union.py  # supplementary: pruning of 14-member committees
```

Outputs are written to `outputs/tables` and `outputs/figures`.

## Running the experiments from scratch

### 1. Tabular committees

Committees of eight learners are trained with five-fold outer cross-validation on the OpenML-CC18 tasks with at most 500
features; out-of-fold predictions come from five-fold inner cross-fitting. The committee files (about 0.5 GB) are written
to `data/` (or to the directory given by the environment variable `FH_COMMITTEES`).

```bash
python scripts/build_committees.py                 # full training folds
python scripts/build_committees.py --n-train 128   # n = 128 regime
python scripts/build_committees.py --n-train 512   # n = 512 regime
python scripts/build_committees.py --alt           # six additional learners (and with --n-train as above)
python scripts/merge_committees.py                 # 14-member committees
```

### 2. Estimates and experiments

For each regime `r` in `n128`, `n512`, `full`:

```bash
python scripts/run_real.py --split all --regime r            # headroom terms and diversity measures
python scripts/run_real.py --split all --regime r --lc       # with learning-curve correction (n128, n512)
python scripts/run_accounting.py --split all --regime r      # gain accounting
python scripts/add_select_cv.py                              # cross-validated selection of the best member
python scripts/unified.py                                    # one consistent set of estimates per instance
python scripts/run_pruning.py --split test --regime full     # pruning (add --family _union for 14 members)
```

Further analyses: `validate_synth_v2.py` (estimator validation), `run_robust.py` (alternative bounds),
`check_circularity.py` (rule-specific baselines), `rq4_accounting.py` (cross-over sizes) and `rq5_selector.py`
(choosing the fusion rule).

The realised test losses of the fusion rules in `data/rule_results/` were computed on the same committees. The classic
and fitted rules are implemented in `src/fh/rules.py`; neural ensemblers use the official implementation with its
default settings; for the pretrained fusion rule the stored test losses are provided.

### 3. Deep committees

```bash
bash scripts/deep_pool.sh                     # trains the 15 members per data set (GPU), resumable
python scripts/deep_analyze.py --dataset cifar10
python scripts/deep_analyze.py --dataset cifar100
python scripts/deep_prune.py --dataset cifar10
python scripts/deep_prune.py --dataset cifar100
python scripts/deep_calib.py
```

`deep_pool_parallel.sh` with `deep_jobs.txt` runs several members in parallel on one GPU.

## Data

All data sets are public: the OpenML-CC18 benchmark suite (downloaded through the `openml` package) and the official
CIFAR-10 and CIFAR-100 releases (downloaded through `torchvision`). Data sets are assigned to the development or test
split by the md5 hash of their name (`src/fh/data.py`); all estimator design choices were made on the development data
sets.

## License

MIT, see `LICENSE`.
