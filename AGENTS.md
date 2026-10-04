# AGENTS.md — handoff for coding agents

Read this first. It describes what this repository is, where the work stands, what
is running, and the rules that apply. Last updated 2026-10-04.

## 1. Project

B.Tech project (NSUT, Dept. of Computer Science and Engineering; supervisor
Dr Ankur Gupta; team Aman Bihari 2023UCA1910, Deepak 2023UCA1913, Deepak
2023UCA1914): **"Topology-Aware Graph Representation for Deep Reinforcement
Learning-Based Routing on MEDA Biochips."** The team intends to publish it.

- **Base paper:** Elfar et al., "Deep reinforcement learning-based approach for efficient
  and reliable droplet routing on MEDA biochips", IEEE TCAD 2023 (cited as `elfar2023`);
  the authors' code is github.com/melfar87/MEDA (`elfarcode`).
- **This repository** reimplements their simulator and their CNN–PPO agent in PyTorch
  (Stable-Baselines3, Gymnasium), then replaces the CNN encoder with a graph encoder.
  The chip is a graph with one node per microelectrode cell (MC) and 8-neighbour edges.
- **Configurations compared**, and only these three, everywhere:
  1. **CNN–PPO baseline:** Table I CNN, 64/128/128 filters, FC 256, 29.7 M parameters,
     on the 30×30 observation.
  2. **GCN + global max pooling:** 3 layers, d = 64, 8.6 k parameters. This is the
     proposed representation.
  3. **Direction-aware GCN + global max pooling:** a relational GCN with its own weight
     matrix per direction, 76 k parameters. It is the ablation of the message-passing layer.
- **Key theory**, verified numerically and part of the report:
  - The isotropic GCN is invariant to the grid's mirror and rotation symmetries for any
    permutation-invariant readout. It therefore cannot tell left from right, and it
    fails to learn routing.
  - The direction-aware GCN is mathematically identical to a 3-layer 3×3 CNN with zero
    padding followed by global max pooling.

## 2. Setup

The working branch is **`meda/gnn/graph_routing`**. A local `main` differs, so check
out this branch:

```bash
git fetch origin meda/gnn/graph_routing
git checkout -B meda/gnn/graph_routing origin/meda/gnn/graph_routing
pip install -e ".[dev]"      # installs the `meda` command
pytest                        # full test suite
```

Main commands:

- `meda train -c configs/training/<cfg>.yaml [--set key=value ...]`
- `meda compare-methods --method LABEL runs/<name> ... --episodes 500 --out results/<dir>`
- `meda validate-graph`
- `meda devices`

`README.md` covers the rest. `FILES.txt` explains every file.

## 3. Repository map

- `src/meda_routing/`
  - `envs/`: the MEDA simulator.
  - `agents/cnn.py` and `agents/gnn.py`: the encoders. `gnn_type` is `gcn` or `dir_gcn`.
  - `agents/graph_readout.py`
  - `training/`: PPO, the learning-rate schedule and config loading. Configs inherit
    from each other through `base:`.
  - `experiments/`: `compare-methods`.
- `configs/training/`
  - Every value is tagged `[PAPER ...]`, `[REF-CODE]`, `[ASSUMED]` or `[METHOD-DOC]`.
    Keep that convention.
  - `paper_30x30_healthy.yaml`: the baseline.
  - `gnn_maxpool_30x30.yaml` and `gnn_dirgcn_30x30.yaml`
  - `*_16x16.yaml`: the CPU study.
  - `paper_<N>x<N>_healthy.yaml` and `gnn_dirgcn_<N>x<N>.yaml` for N = 50, 60, 100, 120:
    the chip-size study.
  - `reference_0825a_30x30.yaml`: mirrors the authors' logged run.
- `docs/`
  - `GNN_METHODOLOGY.md`: the method specification.
  - `IMPLEMENTATION_NOTES.md`: paper vs. code decisions.
  - `RESULTS.md`: validation of the reimplementation.
- `report/`
  - `main.tex`: the mid-semester report; build it with `pdflatex` twice.
  - `make_figures.py`: run `PYTHONPATH=src python report/make_figures.py`.
  - `figures/`
  - `viva/viva_questions.tex`: viva Q&A.
- `kaggle/`: GPU runs on Kaggle (section 5).
- `scripts/reference/`: the test against the authors' original code (section 6).
- `runs/`: gitignored. Only the 16×16 runs are committed: their logs, configs and the
  small GCN models. CNN models are about 26 MB to 0.3 GB, so never commit them.
- `kaggle/output/`: downloaded Kaggle outputs. It is gitignored, but the small result
  files of the 30×30 runs were force-added.

## 4. Results so far (one seed each)

**16×16 study:**

- Runs on CPU, with a smaller CNN (32/64/64, FC 128).
- 40 epochs of 2^13 steps, 300 evaluation jobs, `runs/*_16x16`.

| Method | Success | Mean cycles | Steps to convergence |
|---|---|---|---|
| CNN–PPO | 96.2% | 5.94 | 221 k |
| GCN + max pooling | 4.0% | 25.06 | not reached |
| Direction-aware GCN | 99.4% | 4.89 | 74 k |

**30×30 study:**

- Runs on Kaggle GPU.
- 25 epochs of 2^14 steps, 500 jobs.
- Outputs in `kaggle/output/{cnn,gcn}/`; figure in `report/figures/learning_curves_30x30.png`.

| Method | Success | Mean cycles | Invalid actions per decision | Convergence |
|---|---|---|---|---|
| CNN–PPO | 70.2% | 19.54 | 0.21 | not reached (best epoch 73.8%, still rising) |
| GCN + max pooling | 2.8% | 38.09 | 0.88 | not reached |
| Direction-aware GCN | 100% | 9.87 | 0.000 | 147 k steps |

**Caveat on the 30×30 baseline:**

- The CNN did not finish learning within 25 epochs. The authors' own published 30×30 log
  (`policy/0825a_030x030_E100_NPS64.pickle` in their repository, same CNN and settings)
  shows the same shape: 61% at epoch 20, 96% at epoch 25, converged at epochs 27–29.
- So "100% vs 70%" holds only for a fixed 25-epoch budget. The supportable claim is
  faster convergence, not a higher final success rate.
- **Recommendation:** rerun 30×30 with 40 epochs for both methods.
- `report/main.tex` currently reports the 25-epoch numbers and says the CNN had not
  converged.

## 5. Kaggle workflow (runs on the user's machine)

kaggle.com may be blocked for cloud agents; the user runs the Kaggle CLI locally.

- **Concurrency limit:** at most **2** kernels at a time.
- **Run limit:** each kernel is stopped after 12 h; the GPU quota is about 30 h per week.
- **Kernel template:** `kaggle/run_kernel.py` clones this branch, so **push configs and
  scripts to GitHub before pushing kernels**.
  - It trains with `schedule.checkpoint_every=0`.
  - Kaggle's P100 may be unsupported by current PyTorch; the template falls back to CPU.
- **Push quirk:** `kaggle kernels push` exits 0 even when it fails. Check for
  "successfully pushed".
- **Status quirk:** `kaggle kernels status` can return "Permission 'kernels.get' was
  denied" for a few minutes after a new private kernel is created. That is not a failure.

Scripts:

- **30×30 comparison:**
  - `push_30x30.sh` and `fetch_30x30.sh`.
  - Status: done.
- **Chip-size study** (CNN vs. direction-aware GCN; GCN + max pooling is excluded):
  - `push_sizes.sh` (`SIZES`, `ONLY`, `SEEDS`, `EXTRA`, `WAIT`) and `fetch_sizes.sh`.
  - Status: **not launched yet**.
  - Rough cost of the direction-aware GCN, which grows with the number of MCs: 50×50
    about 1.3 h, 60×60 about 2 h, 100×100 about 5 h, 120×120 about 8 h, near the 12 h
    limit. These are estimates, not measurements.
  - Consider `EXTRA="--set schedule.epochs=40"` for the smaller sizes. Fewer epochs, or
    leaving it out, may be needed for 120×120.
- **Reference test:** `push_reference.sh` and `fetch_reference.sh` (section 6).

## 6. In progress: reference test against the authors' original code

**Goal:** show that the reimplementation trains like the original code, for the paper.

- **`meda-ref-orig-s<k>`:** runs the authors' **unmodified** code at commit `1667016`.
  - It runs in its original stack: Python 3.7, TF 1.15.5, stable-baselines 2.10.1,
    gym 0.18, built by `scripts/reference/setup_original_env.sh` with micromamba from
    conda-forge.
  - `scripts/reference/run_original_0825a.py` loads the argument set from their 0825a
    log and changes only `n_epochs` (40), `seed` and model saving.
  - It runs on CPU, at about 4.5–5 min per epoch on 4 cores.
- **`meda-ref-ours`:** this code with `configs/training/reference_0825a_30x30.yaml`,
  3 seeds, GPU.
- **Smoke test, epoch 1:** both gave 2.5% success, at 59.5 cycles (original) and
  58.6 cycles (ours).
- **Status on 2026-10-04:**
  - `meda-ref-ours` and `meda-ref-orig-s0` are RUNNING on Kaggle.
  - `-s1` and `-s2` still need pushing:
    `ONLY=orig ORIG_SEEDS="1 2" bash kaggle/push_reference.sh`.
  - A local run of the original code was lost when the cloud container restarted.
- **When all four finish:**
  ```bash
  git clone https://github.com/melfar87/MEDA /tmp/MEDA
  AUTHORS_LOG=/tmp/MEDA/policy/0825a_030x030_E100_NPS64.pickle bash kaggle/fetch_reference.sh
  ```
  - This writes `results/reference_test/`: `per_run.csv`, `summary.csv`, `curves.csv` and
    `learning_curves.png`.
  - The metrics are epochs to 90% and 95% success, the convergence epoch (first of three
    evaluations in a row at ≥ 95%), and final success and cycles (mean of the last 5
    epochs).
  - Then write up the result in `docs/RESULTS.md` and the report, and commit the small
    result files.

## 7. Report, slides and viva: rules

- **Overleaf round-trip:** the user edits `report/main.tex` in Overleaf and pastes it
  back. Treat the pasted version as authoritative, apply changes to it, compile twice
  (0 errors, no overfull boxes), and return the full LaTeX plus any new figures.
- **Writing style:** formal, official-report register.
  - Report only the three configurations in section 1.
  - Never mention reproducing or replicating the baseline; cite Elfar et al. instead.
- **Fixed wording:** the supervisor's department is "Dept. of Computer Science and
  Engineering". Use "GCN", not "GNN", in reader-facing text.
- **State results accurately:** single-seed results must say so, and so must the
  unconverged-baseline caveat (section 4).
- **Slides:** the deck lives as a claude.ai artifact, not in this repository.

## 8. Conventions

- Commit and push only to `meda/gnn/graph_routing`. Don't open PRs unless asked.
- No model identifiers in commits or files.
- New experiment configs inherit with `base:` and change only what differs. Every new
  value gets its source tag.
- Keep `FILES.txt` up to date when adding files.
- Never commit large `model.zip` or `best_model.zip` files, or Kaggle checkpoints.
