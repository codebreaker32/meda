# AGENTS.md — handoff for coding agents

Read this first. It describes what this repository is, where the work stands, what
is running, and the rules that apply. Last updated 2026-10-05.

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
- **Key theory** (details and evidence in `docs/PAPER_NOTES.md`, F1 and F3):
  - The isotropic GCN is invariant to the grid's mirror and rotation symmetries (D4) for
    any permutation-invariant readout. It therefore cannot tell left from right, and it
    fails to learn routing.
  - The direction-aware GCN is mathematically identical to a 3-layer 3×3 CNN with zero
    padding followed by global max pooling.
  - Both were checked numerically in an audit on 2026-10-04, but the committed tests cover
    only the left–right mirror with max pooling. `report/main.tex` states neither claim yet.

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
  - `PAPER_NOTES.md`: everything the paper needs (findings with evidence and caveats,
    setup, threats to validity, wording rules, the `main.tex` update checklist).
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

**40-epoch runs (chip-size study, section 5):** CNN vs. direction-aware GCN, 500 held-out
jobs, `results/chip_size/summary.md`.

| Chip | CNN–PPO | Direction-aware GCN |
|---|---|---|
| 30×30 | 87.4%, 14.52 cycles, not converged (best per-epoch 89.4% at epoch 40) | 100%, 9.84 cycles, converged at 147 k steps |
| 50×50 | 44.6%, 41.04 cycles, not converged | 100%, 16.71 cycles, converged at 213 k steps |
| 60×60 | 27.8%, 53.48 cycles, not converged | 99.6%, 20.94 cycles, converged at 213 k steps |
| 100×100 | 15.6%, 86.47 cycles, not converged | 99.8%, 34.31 cycles, converged at 459 k steps; **34 of 40 epochs** (stopped at the 11 h limit) |

At 100×100 the budgets are unequal in the CNN's favour (at epoch 34 its per-epoch success
was 6.4%). Paired tests and the other per-size figures are in `docs/PAPER_NOTES.md` F2.

**Caveat on the 30×30 baseline:**

- The CNN converged neither in 25 nor in 40 epochs, so only fixed-budget claims hold.
  "Faster convergence" is the safe headline.
- The authors' published 30×30 log (`policy/0825a_030x030_E100_NPS64.pickle`) is **not**
  a like-for-like reference. It used the same Table I CNN but different settings: five
  droplet sizes (4×4 … 6×6, no 2×2 or 3×3), `k_max` = W + H = 60, collision marks, and ×0.5
  learning-rate decay only at 100% success.
  - Under those settings our CNN converges at epochs 22–25 (reference test, 3 seeds);
    under the paper settings it does not converge.
  - Log milestones: 61.2% at epoch 20, 96.2% at 25, first ≥ 99% at 27, first 100% at 29;
    converged at 25 by our rule.
- The CNN run did not reproduce at a fixed seed: two seed-0 runs, in different Kaggle
  software environments (Python 3.12 vs. 3.13, unpinned packages), give 67.2% vs. 72.4% at
  epoch 25. The cause is not isolated. The GCN rerun matched exactly.
- `report/main.tex` still reports the 25-epoch numbers. The update checklist is in
  `docs/PAPER_NOTES.md` §9.

## 5. Kaggle workflow (runs on the user's machine)

kaggle.com may be blocked for cloud agents; the user runs the Kaggle CLI locally.

- **Concurrency limit:** at most **2** GPU kernels at a time, and CPU-only kernels seem to
  count toward it. On 2026-10-04 one GPU and three CPU-only kernels ran together, CPU
  pushes went through, but every GPU push was refused ("Maximum batch GPU session count
  of 2 reached") although the GPU quota showed a single GPU session.
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
- **Chip-size study** (CNN vs. direction-aware GCN on 30, 50, 60, 100 and 120; GCN + max
  pooling is excluded):
  - `push_sizes.sh` (`SIZES`, `ONLY`, `SEEDS`, `EPOCHS` = 40, `EXTRA`, `OTHER_GPU`, `FETCH`)
    and `fetch_sizes.sh`.
  - Kernels `meda-size-<N>x<N>-cnn` and `meda-size-<N>x<N>-dirgcn`: 40 epochs, 1 seed. The
    30×30 pair is the 40-epoch rerun of section 4; the 25-epoch kernels `meda-30x30-*` are
    left untouched.
  - `push_sizes.sh` is a queue, smallest chip first. It keeps at most 2 GPU kernels
    running, counting `meda-ref-ours`, and downloads and compares each size once both of
    its kernels finish. Outputs: `kaggle/output/chip_size/<N>x<N>-{cnn,dirgcn}/`,
    `results/chip_size/<N>x<N>/`, and `results/chip_size/summary.csv` (one row per size and
    method, with the epochs trained).
  - Training stops by 11 h (`schedule.max_hours`, set by `run_kernel.py`). A run stopped
    that way is flagged in `summary.json` and `summary.csv`, and the report must say so.
  - Cost on a T4, seconds per epoch (training plus the 500-job evaluation):

    | Method | 30×30 | 50×50 | 60×60 | 100×100 |
    |---|---|---|---|---|
    | CNN | 59–60 | 66 | 73 | 94 |
    | Direction-aware GCN | 67 | 204 | 264 | 1,156 |

    Whole kernels: CNN 0.70, 0.78, 0.86 and 1.09 h; GCN 0.77, 2.29, 2.97 and 10.96 h (the
    last stopped by the 11 h limit). The GCN's time grew faster than the MC count from 60×60
    to 100×100. Projected for 120×120: 1,665–1,957 s per epoch, so only 20–23 epochs fit in
    11 h.
  - Mean `k_max` of the held-out jobs: 38.7 cycles at 30×30, 65.0 at 60×60, 115.4 at
    120×120.
  - Status on 2026-10-05 15:00 UTC:
    - 30×30, 50×50, 60×60 and 100×100 are done and compared. No kernel is running.
    - **120×120 is deferred.** 24.5 h of the 30 h weekly GPU quota are used; it resets on
      2026-10-10 00:00 UTC. 120×120 needs about 11–12.5 GPU-hours.
    - The epoch budget for 120×120 is the user's decision (`docs/PAPER_NOTES.md` §7, item
      9): `EPOCHS=20` gives both methods the same budget but the GCN may not converge
      (it needed 28 epochs at 100×100); `EPOCHS=40` repeats the 100×100 situation. Launch:
      `EPOCHS=<n> SIZES=120 nohup bash kaggle/push_sizes.sh >> kaggle/build/push_sizes.log 2>&1 &`
    - The scheduler pauses while the PC sleeps and stops if WSL restarts (it did on
      2026-10-04 at 17:01). Pushed kernels keep running.
    - Rerun the scheduler with `SIZES` set to the sizes not yet pushed; use `OTHER_GPU`
      for kernels already running.
    - As each size finishes, `fetch_sizes.sh` regenerates `results/chip_size/summary.csv`
      and `summary.md`.
- **Reference test:** `push_reference.sh` and `fetch_reference.sh` (section 6).

## 6. Reference test against the authors' original code

**Goal:** show that the reimplementation trains like the original code, for the paper.

- **`meda-ref-orig-s<k>`:** runs the authors' **unmodified** code at commit `1667016`.
  - It runs in its original stack: Python 3.7, TF 1.15.5, stable-baselines 2.10.1,
    gym 0.18, built by `scripts/reference/setup_original_env.sh` with micromamba from
    conda-forge.
  - `scripts/reference/run_original_0825a.py` loads the argument set from their 0825a
    log and changes only `n_epochs` (40), `seed`, model saving and output names.
  - It runs on CPU: 299, 466 and 467 s per epoch for seeds 0, 1 and 2.
  - **Network caveat:** the authors' `my_net.py` at `1667016` builds a 32/64/64 + FC 128 CNN
    (about 7.43 M parameters), not the Table I network our side trains (29.7 M). Its
    learning-rate decay is ×0.7, against ×0.5 on our side.
- **`meda-ref-ours`:** this code with `configs/training/reference_0825a_30x30.yaml`,
  3 seeds, GPU.
- **Smoke test, epoch 1:** both gave 2.5% success, at 59.5 cycles (original) and
  58.6 cycles (ours).
- **Status: done (2026-10-04).** All four kernels finished, and the outputs are in
  `kaggle/output/reference/` and `results/reference_test/`. The write-up is in
  `docs/RESULTS.md` §7 and `docs/PAPER_NOTES.md` F4.

  | | First ≥ 95% success (epoch) | Converged (epoch) | Final success | Final cycles |
  |---|---|---|---|---|
  | Original code | 25.7 ± 3.1 | 25.7 ± 3.1 | 99.81% | 7.95 |
  | Ours | 20.3 ± 0.6 | 23.3 ± 1.5 | 99.05% | 8.42 |

  - With 3 seeds per side, no difference survives correction.
  - A clean comparison needs a rerun of our side with the network and decay rule of
    `1667016` (about 2 GPU-hours; `docs/RESULTS.md` §7).
- A local run of the original code was lost when the cloud container restarted.
- **To redo the comparison:**
  ```bash
  git clone https://github.com/melfar87/MEDA /tmp/MEDA
  AUTHORS_LOG=/tmp/MEDA/policy/0825a_030x030_E100_NPS64.pickle bash kaggle/fetch_reference.sh
  ```
  It writes `results/reference_test/` (`per_run.csv`, `summary.csv`, `curves.csv` and
  `learning_curves.png`). The metrics are epochs to 90% and 95% success, the convergence
  epoch (first of three evaluations in a row at ≥ 95%), and final success and cycles
  (mean of the last 5 epochs).

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
