# Paper notes

The working record for the paper. It holds every result, setting and caveat the paper needs,
each with the file it comes from, plus the wording rules and a checklist for updating
`report/main.tex`. Every number was re-derived from its primary source (CSV, JSON, pickle or
code). The derivations are summarised in "Source" columns.

Last updated 2026-10-06, 12:25 UTC. All results are from **one training seed per
configuration** unless a row says otherwise. Only the three configurations of AGENTS.md §1
are reported:

- the CNN–PPO baseline of Elfar et al. [elfar2023], our implementation;
- GCN + global max pooling;
- direction-aware GCN + global max pooling.

## 0. Status

| Study | Status | Where |
|---|---|---|
| 16×16, CPU, 3 configurations | done | `results/tables/`, `runs/*_16x16/` |
| 30×30, 25 epochs, 3 configurations | done | `kaggle/output/{cnn,gcn}/` |
| 30×30, 40 epochs, CNN vs. direction-aware GCN | done | `results/chip_size/30x30/` |
| 50×50, 40 epochs | done | `results/chip_size/50x50/` |
| 60×60, 40 epochs | done | `results/chip_size/60x60/` |
| 100×100, 40 epochs planned | done; the GCN stopped at the 11 h limit after 34 epochs | `results/chip_size/100x100/` |
| 120×120, 40 epochs planned | running since 2026-10-06 12:17 UTC (second Kaggle account, §6). The GCN will most likely stop at the 11 h limit after 20–23 epochs (F5; §7, item 9). | `kaggle/build/push_sizes.log` |
| Reference test: original code vs. our implementation | done, 3 seeds each | `results/reference_test/`, `docs/RESULTS.md` §7 |

`results/chip_size/summary.md` is regenerated automatically as each chip size finishes. It
is the live results table for the chip-size study.

## 1. Findings the paper can claim

### F1. GCN + global max pooling cannot learn routing, because of grid symmetry

**Claim.** The isotropic GCN treats every neighbour alike. Every symmetry of the grid (the 8
elements of D4 on a square chip) is a graph automorphism, so the encoder is
permutation-equivariant and any permutation-invariant readout (max, mean or sum) is invariant.
A job and its mirror image or rotation therefore get the same action distribution, although
their correct moves differ.

**Evidence.**
- Held-out success, single seed: **4.0%** on 16×16 (20/500, Wilson 95% CI 2.6–6.1%) and
  **2.8%** on 30×30 after 25 epochs (14/500). Its per-epoch success never exceeded 5.0%
  (16×16) or 3.4% (30×30). Sources: `results/tables/main_comparison.csv`;
  `kaggle/output/gcn/results/gnn_maxpool_30x30/tables/`.
- Invalid actions per decision: 0.88. The 14 jobs it solved at 30×30 are short (6.0
  cycles on average), and the direction-aware GCN solved each of them in exactly the same
  number of cycles.
- The trained 30×30 model chooses action **E in all 500 held-out jobs** and in all 500
  audit start states, and its embedding is identical (to float32 rounding) under left–right
  and up–down mirroring, 90° and 180° rotation, and transposition. The 16×16 model chooses
  N in 390 of the 500 held-out jobs (386 of the 500 audit start states).
  These are audit computations on the committed models; no committed script exists yet.
- *Audit start states* (used in F1, F2 and F3 wherever named): 500 single-job resets,
  `env.reset(seed=20000+i)` for i = 0–499. They are **not** the held-out job set, which is
  drawn from 8 seeded environment streams (job seed 20000). Cite held-out figures where both
  are given.
- Unit test `test_isotropic_gcn_with_max_pooling_cannot_tell_a_job_from_its_mirror_image`
  (`tests/test_graph_readout.py`) covers only the left–right mirror, with max pooling and
  random weights. An audit check covering all 7 non-identity D4 maps and max, mean and sum
  pooling, at 16×16 and 30×30, gave a relative embedding difference ≤ 3.0e-7.

**A bound the paper can state.** Take a job whose goal lies straight along an axis from
the droplet. Its 8 D4 images need the 4 cardinal actions, each twice. A D4-invariant policy
gives all 8 images the same action distribution p, so averaged over the orbit it picks the
goal direction with probability (2p_N + 2p_S + 2p_E + 2p_W)/8 ≤ 1/4.

**The trained GCN is a constant policy.** It always chooses E, so its 45.8% rate of
distance-reducing actions at 30×30 is exactly the share of audit start states whose goal
lies east of the droplet (229/500). On the held-out jobs the share is 225/500 (45.0%); the
sampler's base rate is 46.9%.

**Precise version, for the methods text.** This was tested empirically on one geometry, not
proved in general:
- Geometry: a 60×60 chip and a 4×4 droplet, with the goal 12 MCs east vs. 12 MCs west.
- With the trained 30×30 GCN weights and max pooling, the two jobs get the same embedding
  (to float64 rounding) whenever both routing zones stay at least 1 MC from the chip edge.
  With mean and sum pooling the threshold is 4 MCs.
- With random weights (normal, σ = 0.5, biases included) the threshold is 6 MCs for all
  three readouts. The paper must say which weights a threshold refers to.
- Only the chip boundary breaks the symmetry. Biases give zero-feature nodes embeddings that
  depend on their distance to the edge, and that dependence is itself mirror-symmetric.
- The routing zone touches the chip edge in 480 of the 500 held-out 30×30 jobs (479 of the
  500 audit start states), so a boundary cue usually exists. The trained GCN still collapsed to a constant action.

### F2. The direction-aware GCN learns faster and routes better than the CNN baseline at equal (or, at 100×100, smaller) step budgets

Held-out evaluation of the final model: 500 jobs, job seed 20000, deterministic policy, the
same jobs for every method. The CNN baseline did not converge at any chip from 30×30 to
100×100 within the budget, so those rows are fixed-budget comparisons (at 16×16 both methods
converged). Steps per epoch are 2^14 = 16,384 at every size from 30×30 up.

| Chip, budget | CNN–PPO baseline | Direction-aware GCN | Source |
|---|---|---|---|
| 16×16, 40 × 2^13 steps (CPU) | 96.2%, 5.94 cycles, converged 221,184 steps | 99.4%, 4.89 cycles, converged 73,728 steps (3.0× fewer) | `results/tables/` |
| 30×30, 25 × 2^14 steps | 70.2%, 19.54 cycles, not converged (best per-epoch 73.8%) | 100%, 9.87 cycles, converged 147,456 steps | `kaggle/output/{cnn,gcn}/results/` |
| 30×30, 40 × 2^14 steps | 87.4%, 14.52 cycles, not converged (best per-epoch 89.4%, at epoch 40) | 100%, 9.84 cycles, converged 147,456 steps (epoch 9) | `results/chip_size/30x30/` |
| 50×50, 40 × 2^14 steps | 44.6%, 41.04 cycles, not converged (43.0% at epoch 40) | 100%, 16.71 cycles, converged 212,992 steps (epoch 13) | `results/chip_size/50x50/` |
| 60×60, 40 × 2^14 steps | 27.8%, 53.48 cycles, not converged (30.6% at epoch 40) | 99.6%, 20.94 cycles, converged 212,992 steps (epoch 13) | `results/chip_size/60x60/` |
| 100×100, CNN 40 × 2^14 steps, GCN 34 × 2^14 (11 h limit) | 15.6%, 86.47 cycles, not converged (14.4% at epoch 40, 6.4% at epoch 34) | 99.8%, 34.31 cycles, converged 458,752 steps (epoch 28) | `results/chip_size/100x100/` |

- **Paired tests on identical jobs** (exact McNemar test on success; two-sided sign test on
  the cycles of jobs both methods solved):

  | Chip, budget | Jobs only the GCN solved / only the CNN solved | McNemar p | Both solved: GCN vs. CNN cycles (jobs) | GCN fewer / more cycles | Sign p |
  |---|---|---|---|---|---|
  | 16×16 | 19 / 3 | 0.00086 | — | — | — |
  | 30×30, 25 epochs | 149 / 0 | 2.8e-45 | 9.03 vs. 11.81 (351) | — | 4.3e-66 |
  | 30×30, 40 epochs | 63 / 0 | 2.2e-19 | 9.25 vs. 11.08 (437) | 291 / 10 | 7.4e-73 |
  | 50×50 | 277 / 0 | 8.2e-84 | 13.50 vs. 20.96 (223) | 193 / 2 | 7.6e-55 |
  | 60×60 | 360 / 1 | 1.5e-106 | 16.69 vs. 25.21 (138) | 110 / 2 | 2.4e-30 |
  | 100×100 | 421 / 0 | 3.7e-127 | 32.06 vs. 52.94 (78) | 68 / 1 | 2.4e-19 |

  Source: `results/chip_size/<N>x<N>/logs/evaluation_jobs.csv`. The p values are far below
  any threshold, but they rest on one training seed per method: they show the difference
  between these two trained policies, not between the methods in general.
- **Size of the encoders.** The GCN encoder has 75,648 parameters, the Table I CNN encoder
  29,714,688: 392.8× fewer.
- **The baseline does not converge on the paper's job distribution.** At 30×30 its
  per-epoch success over epochs 20/25/30/35/40 was 67.2/72.4/75.8/83.2/89.4%. At 40 epochs
  the GCN has used 4.4× fewer steps to converge (147,456 vs. 655,360+). At epoch 40 the
  CNN's per-epoch success was 43.0% (50×50), 30.6% (60×60) and 14.4% (100×100).
- **The GCN needs more steps on larger chips, but it converges at every size.** It
  converged at epoch 9 (30×30), 13 (50×50 and 60×60) and 28 (100×100).
- **The gap widens with chip size.** Held-out success of the CNN falls from 87.4% (30×30)
  to 44.6%, 27.8% and 15.6% (50×50, 60×60, 100×100); the GCN stays at 99.6–100%.
- **Small droplets are the hardest.** Held-out success of the 30×30 CNN by droplet size
  after 40 epochs: 64.3% (36/56, 2×2), 75.0% (42/56, 3×3), 92.5% (359/388, 4×4–6×6).
  After 25 epochs: 41.1%, 50.0%, 77.3% (300/388).

Caveats, all of which must appear in the paper:
- Single seed. The CNN run did not reproduce at a fixed seed: two seed-0 runs read 67.2% vs.
  72.4% at epoch 25 and differ from epoch 1. They ran in different Kaggle software
  environments (Python 3.12 vs. 3.13, unpinned packages, different commits), so the cause is
  not isolated; GPU (cuDNN) nondeterminism is plausible but unproven. The GCN rerun, across
  the same two environments, matched its earlier run exactly for epochs 1–25.
- The CNN baseline never converged at any size from 30×30 (25 and 40 epochs) to 100×100,
  so only fixed-budget claims hold there; "faster convergence" is the safe headline.
- At 100×100 the budgets are unequal: the GCN stopped at the 11 h limit after 34 epochs
  (557,056 steps), the CNN trained all 40 (655,360). The difference favours the CNN; at
  epoch 34 the CNN's per-epoch success was 6.4% (best over epochs 1–34: 11.4%). The table
  must say "34 epochs (11 h limit)" for that row.
- Above 30×30 the CNN sees an observation resampled to 30×30 (`cv2.INTER_AREA`, (N/30)² MCs
  per pixel), while the GCN sees the native grid. Input resolution is confounded with the
  encoder there.
- The GCN is a fully convolutional network with global max pooling (F3). Readout, width
  (64/64/64 vs. 64/128/128) and embedding size (64 vs. 256) all differ from the baseline.
- Learning-rate asymmetry: the paper's decay rule (×0.7 after an epoch above 99%) fired only
  for the GCN. The CNN trained at 3.5e-4 throughout.
- Healthy chips only (§3): degradation-aware routing is not tested.

### F3. The direction-aware GCN is exactly a 3-layer 3×3 CNN with zero padding and global max pooling

**Statement.** Each direction-aware layer is h'_i = ReLU(W_0 h_i + b + Σ_r W_r h_{i−o_r}),
with one relation per neighbour direction, unit edge weights and no relation bias. It equals
a 3×3, stride-1 convolution with zero padding of 1:
- W_0 is the centre tap;
- relation r with offset (dx, dy) is the tap at (row 1−dy, column 1−dx);
- a neighbour missing at the chip edge contributes nothing, which is exactly zero padding.

The parameter counts agree: 1,792 + 36,928 + 36,928 = 75,648 = Conv2d(3,64,3) +
2 × Conv2d(64,64,3).

**Numerical check (audit).** The trained 30×30 weights were copied into Conv2d plus
AdaptiveMaxPool2d. Over the 500 audit start states the graph embeddings differ by at most 6.1e-5 in
float32 and 6.4e-14 in float64; random weights on 11×7 and 120×120 grids give similar
results. No committed test exists yet, although AGENTS.md and the viva say "verified
numerically".

**Trained behaviour.**
- The trained 30×30 direction-aware GCN separates mirrored and rotated states (no identical
  embeddings under mirroring or rotation).
- Its greedy action transforms with the job (audit start states): 475/500 consistent under
  the left–right mirror,
  477/500 under the up–down mirror.
- It mostly moves diagonally (421/500 audit start states).
- Global max pooling can still merge a job and its transpose in rare corner cases (5/500
  audit start states; in all five the routing zone reaches a chip corner, and in four the
  droplet or the goal sits at the corner).

**Framing.**
- The graph view explains why isotropic message passing fails and which inductive bias
  routing needs (direction-specific weights, translation equivariance, a readout
  independent of size).
- The resulting encoder has weights independent of chip size and 393× fewer parameters than
  the Table I CNN.
- Cite the related work the viva names but `main.tex` does not: Grid-to-Graph (Jiang et
  al., AAMAS 2021), relational deep RL (Zambaldi et al., ICLR 2019) and Directional Graph
  Networks (Beaini et al., ICML 2021). The layer is the R-GCN of Schlichtkrull et al.
  (2018) with c_{i,r} = 1.

### F4. Our implementation trains comparably to the authors' original code (reference test)

Full write-up in `docs/RESULTS.md` §7. Three seeds per side, the settings of the authors'
logged run 0825a, 40 epochs:

| | first ≥ 95% (epoch) | converged (epoch) | success, epochs 36–40 | cycles, epochs 36–40 |
|---|---|---|---|---|
| Authors' original code (`1667016`) | 25.7 ± 3.1 | 25.7 ± 3.1 | 99.81 ± 0.15% | 7.95 ± 0.61 |
| Our implementation | 20.3 ± 0.6 | 23.3 ± 1.5 | 99.05 ± 0.26% | 8.42 ± 0.14 |

Welch tests: epoch first ≥ 95%, p = 0.089; convergence, p = 0.32; final success, p = 0.019
(0.077 after Holm correction); final cycles, p = 0.31. With n = 3 per side, no exact
permutation test can go below p = 0.10.

**Caveats the paper must state.**
- **Network mismatch.** The authors' code at `1667016` builds a 32/64/64 + FC 128 CNN
  (about 7.43 M parameters); our side trained the Table I CNN (29.7 M), which is also the
  network of the published 0825a model.
- **Learning-rate decay mismatch.** ×0.7 in the code vs. ×0.5 on our side, active only after
  the first 100% epoch.
- **Evaluation protocol.** We use a fixed set of 500 jobs; the original code draws fresh
  jobs and counts the first 500 to finish (about −0.1 cycles of bias, an audit simulation
  estimate with no committed script).
- **Timeouts.** 60 cycles and truncation on our side, 61 cycles and terminal in the original.
- **The 0825a settings are easier than the paper's job distribution** (§4). Do not use the
  reference test or the authors' log to predict the baseline's convergence in our 30×30
  study.

**Suggested wording.** "To verify our implementation of the CNN–PPO baseline of Elfar et
al. [elfar2023], we trained it and the authors' original implementation [elfarcode]
(commit 1667016, unmodified) with the settings of the authors' logged 30×30 run for 40
epochs of 2^14 environment steps, evaluating the deterministic policy on 500 routing jobs
after every epoch (three seeds per implementation). The two implementations reached the
convergence criterion at statistically indistinguishable points (epoch 23.3 ± 1.5 vs.
25.7 ± 3.1; Welch's t-test, p = 0.32) and ended at 99.05 ± 0.26% and 99.81 ± 0.15% success.
The original implementation at this commit uses a smaller CNN (32/64/64 filters, FC 128)
and a learning-rate decay factor of 0.7, so the comparison checks the training pipeline
rather than an identical network."

**Recommended follow-up (about 2 GPU-hours).** Rerun our side with the network and decay
rule of `1667016`:

```
--set agent.extractor_kwargs="{channels: [32, 64, 64], hidden_dim: 128}" --set schedule.lr_decay=0.7 --set schedule.lr_min=1.0e-6
```

This removes the main confound.

### F5. Compute scales with the number of MCs for the GCN; the CNN's cost is fixed by resampling

- Seconds per epoch on a T4 (2^14 training steps plus the 500-job evaluation):

  | Method | 30×30 | 50×50 | 60×60 | 100×100 |
  |---|---|---|---|---|
  | Direction-aware GCN | 67.0 | 204.0 | 264.4 | 1,156.1 |
  | CNN | 59.3–60.2 | 66.2 | 72.9 | 93.8 |
  | GCN + max pooling | 37.4 | — | — | — |

  Means over all epochs, from `progress.csv` of each run.
  - Up to 60×60 the GCN's epoch time grows in proportion to the number of MCs (×3.95 for
    ×4.0 more MCs, 30×30 to 60×60). From 60×60 to 100×100 it grows faster (×4.37 for ×2.78
    more MCs). The cause has not been measured; training and evaluation time are not logged
    separately.
  - The CNN's network cost is fixed by the 30×30 resampling, yet its epoch time still grows
    ×1.58 from 30×30 to 100×100 (simulation and longer evaluation episodes, presumably; not
    measured).

- Analytic multiply-accumulates per forward pass:
  - CNN: 230.1 M at every size (30×30 input).
  - Direction-aware GCN: 19.7 M (16×16), 69.2 M (30×30), 192.4 M (50×50), 277.0 M (60×60),
    769.7 M (100×100), 1,108.4 M (120×120).
  - The GCN overtakes the CNN between 54×54 and 55×55 (N ≈ 2,990 MCs).
  - At 16×16 the baseline was the smaller CNN (16.5 M); the direction-aware GCN needs 1.19×
    that.
- Batch-1 inference on the T4 is dominated by overhead: 1.13 ms (CNN) vs. 1.41–1.50 ms
  (GCN) at 30×30, about the same at 50×50. Do not present it as a compute measure. The
  local-CPU inference numbers in `summary.csv` come from a different, uncontrolled machine.
- Projection for 120×120, from the measured 100×100 time: 1,665 s per epoch if the time is
  linear in the MCs, 1,957 s if the 60×60 → 100×100 power law (exponent 1.44) holds. Only
  20–23 epochs fit in the 10.99 h training budget, against the 28 epochs the GCN needed to
  converge at 100×100. (The projection from 30×30 and 50×50 alone, 1,175–1,375 s, was too
  low: it predicted 820–930 s at 100×100.)

## 2. Method facts

**Graph.**
- One node per MC, indexed y·W + x. Node features are the three baseline observation
  channels, copied without renormalisation: health H/4 inside the routing zone and 0
  outside, a droplet indicator and a goal indicator.
- 8-neighbour edges are stored in both directions, without self-loops (the GCN adds its own)
  and without edge attributes.
- Number of directed edges: |E| = 2[(W−1)H + W(H−1) + 2(W−1)(H−1)] = 4(n−1)(2n−1) for an
  n×n chip. Degrees are 3 (corner), 5 (border) and 8 (interior).
- Feature parity with the CNN observation is tested (`test_node_features_are_the_baseline_observation`).
  On 30×30 the two observations are identical
  (`test_native_graph_observation_equals_the_baseline_image_on_30x30`).

| Chip | Nodes | Directed edges |
|---|---|---|
| 16×16 | 256 | 1,860 |
| 30×30 | 900 | 6,844 |
| 50×50 | 2,500 | 19,404 |
| 60×60 | 3,600 | 28,084 |
| 100×100 | 10,000 | 78,804 |
| 120×120 | 14,400 | 113,764 |

**Layers.**
- GCN: H' = ReLU(D^−½(A+I)D^−½ H W + b). The degree includes the self-loop, so the
  coefficients are 1/√(d_i d_j) with d ∈ {4, 6, 9}. In the interior the layer is a 3×3
  convolution whose 9 taps all equal W/9, a rotation- and reflection-invariant kernel.
- Direction-aware GCN: see F3.
- Both use 3 layers, d = 64 and ReLU, then global max pooling to a 64-dimensional embedding.
- Actor and critic are linear heads on the shared encoder (`net_arch` pi = [], vf = []),
  as for the CNN, so the encoder is the only network difference.

**Receptive field.** 3 hops, i.e. a 7×7-MC block, at every chip size. In 75.8% of the 500
held-out 30×30 jobs (79.2% of the audit start states), no node's receptive field contains
both the droplet and the goal (Chebyshev gap > 6 MCs). An audit probe suggests the trained
direction-aware GCN (25-epoch 30×30 model) uses the edge of the routing-zone mask as a
directional cue. Its greedy action reduces the Manhattan distance in 500/500 held-out jobs
(499/500 audit start states). Setting the health channel to 0.75 over the whole chip, which
removes the mask, lowers this to 417/500 (418/500); other uniform values from 0.25 to 1.0
give 83–93%. No committed script exists. (An earlier draft quoted a drop to 94.2%; that
figure could not be reproduced and must not be used.)

**Parameter counts.** The paper should state whether a count is encoder-only or the whole
policy. AGENTS.md and `main.tex` quote encoder counts; the training logs print totals.

| Configuration | Encoder | Heads | Total (logged) |
|---|---|---|---|
| Table I CNN, 30×30 input | 29,714,688 (FC layer: 29,491,456 = 99.25%) | 2,313 | 29,717,001 |
| Smaller CNN, 16×16 study (32/64/64, FC 128) | 2,153,600 | 1,161 | 2,154,761 |
| GCN + max pooling | 8,576 | 585 | 9,161 |
| Direction-aware GCN + max pooling | 75,648 | 585 | 76,233 |

GCN parameters do not depend on chip size. A Table I FC layer on a native 120×120 grid
would need 471,859,456 parameters.

## 3. Experimental setup

Values and source tags come from `configs/training/paper_30x30_healthy.yaml` and
`src/meda_routing/training/config.py`. Tags: [PAPER] = elfar2023, [REF-CODE] = elfarcode,
[ASSUMED] = our choice.

| Item | Value | Tag |
|---|---|---|
| Chip | 30×30 (also 16, 50, 60, 100, 120 in the size study) | PAPER (30) |
| Droplet sizes | 2×2, 3×3, 4×4, 4×5, 5×4, 5×5, 5×6, 6×5, 6×6, one agent for all | PAPER |
| Job sampling | stratified, edge weight 0.2 (62% of droplets touch an edge at 30×30); start = goal redrawn | REF-CODE |
| Routing zone | bounding box of start and goal + 3 MCs, clipped to the chip | REF-CODE |
| Episode limit | k_max = ⌈α(W_h + H_h)⌉, α = 1 (paper: α ∈ [1, 2]); mean 38.7 cycles on the 30×30 held-out jobs (38.6 on the per-epoch evaluation jobs) | ASSUMED (α) |
| Timeouts | truncation, bootstrapped by PPO (PPO2 treated them as terminal) | ASSUMED |
| Actions | 8 directions; step size from Algorithm 1, ⌊w/2⌋ per axis, never overshooting | PAPER |
| Movement | frontier model of the reference code; success probability = mean D of the frontier | REF-CODE |
| Degradation | D = τ^(n/c), τ ~ U[0.5, 0.7], c ~ U[500, 800] per MC; 2-bit health H = min(⌊4D⌋, 3) | PAPER / REF-CODE |
| Chips | healthy: no faults, no hidden defects, zero initial wear; collision marks off | ASSUMED |
| Reward | +0.5Δd on progress, otherwise 0.8Δd − 1; +100 at the goal; −1 for an invalid action | REF-CODE |
| Observation | 3 × 30 × 30 (health in the zone, droplet, goal); larger chips resampled with `cv2.INTER_AREA`; GCN native | PAPER |
| CNN | Table I: 3×3 convolutions 64/128/128, **stride 1, SAME padding**, FC 256 | PAPER + REF-CODE |
| PPO | 8 environments, n_steps 64, batch 32, 4 epochs, γ 0.99, λ 0.95, entropy 0.01, clip 0.2, value clip 0.2, gradient norm 0.5 | PAPER (8 envs) / REF-CODE |
| Value-loss coefficient | 0.25, PPO2's 0.5 translated to SB3; **0.5 in the 16×16 study** | REF-CODE / ASSUMED |
| Learning rate | 3.5e-4, ×0.7 after an epoch with > 99% success, floor 1e-6; within an epoch × √(remaining fraction) | PAPER / REF-CODE |
| Epoch | 2^14 environment steps (2^13 in the 16×16 study) = 32 PPO updates, 2,048 gradient steps | PAPER |
| Per-epoch evaluation | 500 jobs (300 in the 16×16 study), deterministic, seed 10000, the same jobs every epoch. Used for learning-rate decay, `best_model.zip` and convergence. | PAPER (500) |
| Held-out evaluation | final `model.zip`, 500 jobs, seed 20000, deterministic, identical jobs for every method | ASSUMED |
| Convergence | first of 3 consecutive per-epoch evaluations ≥ 95%, reported at the start of the window | ASSUMED |
| Seeds | 1 (seed 0); the paper repeats 5 times | ASSUMED |

**Disclosures for the methods section.**
1. **Stride.** Table I prints stride 3. We use stride 1, because the authors' saved weights
   have an FC input of 115,200 = 128 × 30 × 30. Neither the stride in Table I nor the FC
   input was re-checked in the audit (the saved model is an unfetched Git LFS object; its
   118,952,022 bytes are consistent with about 29.7 M float32 parameters). Re-check both
   before citing.
2. **Healthy chips.** Over an episode D ≥ 0.5^(60/500) = 0.920, so the sensed health stays
   at 3/3 (0.75) everywhere. The health channel acts only as a routing-zone mask, and
   movement is practically deterministic: 0 of the 4,824 shortest-path moves on the 500
   held-out 30×30 jobs failed.
   Degradation-aware routing, which the introduction motivates, is not exercised.
3. **Value loss.** SB3's clipped value loss uses only the clipped prediction; PPO2 took the
   maximum of the clipped and unclipped losses.
4. **Held-out jobs** are drawn from the same distribution with a different seed and are not
   guaranteed disjoint from training jobs. An earlier audit found that at least 101 of the
   500 16×16 held-out jobs also appear in training; this was not re-verified (it needs the
   training job stream rebuilt).
5. **A cycle reference exists.** The adaptive shortest-path router needs 9.648 cycles on the
   500 held-out 30×30 jobs (100% success; median 8) and 4.644 at 16×16. On healthy chips
   this is a lower bound for any policy that routes every job. The GCN's 9.84 cycles at
   30×30 is within 2% of it. On the jobs it solves, the CNN needs 11.08 cycles after 40
   epochs against an analytic minimum of 9.09 on those jobs, 22% above it (25 epochs: 11.81
   vs. 8.87, 33% above).

## 4. Validation of the simulator and the baseline

1. **The authors' trained agent in our simulator** (`docs/RESULTS.md` §1): 100% of 300 jobs,
   10.00 cycles, score 110.55.
   - Its own training log shows 99.8–100% success at about 10.5 cycles; the frozen-policy
     mean over epochs 41–100 is 10.57.
   - The shortest path needs 7.07 cycles on the same jobs.
   - No primary output file is stored. Rerun `scripts/evaluate_reference_model.py` and
     commit its output before citing it.
2. **Unit tests against the reference code.**
   - The movement model (`_updatePattern`) matches exactly on 2,400 cases, and the reward on
     1,500 random steps.
   - Example 1 of [elfar2023] is checked by a test case.
   - Source: `tests/test_movement.py`, `tests/test_env.py`; Example 1 is in
     `tests/test_geometry_actions.py`.
3. **Training-level reference test:** F4.
4. **Why the authors' log is not a reference for our 30×30 study.** The 0825a run used five
   droplet sizes (4×4 … 6×6), k_max = W + H = 60, collision marks, and ×0.5 decay only at
   100%. The paper defaults use nine sizes including 2×2 and 3×3 (which move 1 MC per step),
   a hazard-based k_max (mean 38.6 on the per-epoch evaluation jobs) and no marks.
   - Under 0825a settings our CNN converges at epochs 22–25 (3 seeds); under paper settings
     it reaches 89.4% after 40 epochs (1 seed).
   - On the per-epoch evaluation jobs (seed 10000), the shortest path needs 7.34 cycles
     under 0825a settings and 9.65 under paper settings. The mean per-job ratio
     k_max / optimal is 12.0 vs. 5.7 (ratio of the means: 8.2 vs. 4.0).
   - Which of the differences slows learning has not been isolated.
   - The authors' log milestones: 61.2% at epoch 20, 96.2% at 25, first ≥ 99% at 27, first
     100% at 29, converged at 25 by our rule. AGENTS.md's "27–29" refers to the 99% and 100%
     milestones.

## 5. Compute and platform

- **30×30 and chip-size runs.** Kaggle kernels with 2 visible Tesla T4 GPUs (14.6 GiB each),
  of which training used one (cuda:0), and 4 logical CPU cores. Python 3.12 (25-epoch
  runs) and 3.13 (chip-size runs). Packages were unpinned and the commit hash was not
  logged; the 25-epoch runs most likely used `9189813`. The chip-size kernels were all
  pushed from 13:30 UTC on 2026-10-04 and clone the branch head, so they used `a7fa1b8` or
  a later docs-only commit; the training code is identical to `d428c78`.
- **16×16 runs.** A cloud container on CPU (Python 3.11.15, SB3 2.9.0, PyTorch 2.14.0 CPU;
  the only source is `report/main.tex` Table 5, no run log records it). Its epoch times vary
  widely; the cause (probably CPU sharing) was not recorded.
- **Original code.** CPU-only Kaggle kernels, Python 3.7 / TF 1.15.5 / stable-baselines
  2.10.1 / gym 0.18.0. The authors declared TF 1.14.0. 299, 466 and 467 s per epoch.
- **Report Table 5 is wrong.** It lists an NVIDIA A100 training server, but no reported
  result was produced on an A100.
- **Measured kernel times so far:**

  | Kernel | 30×30 | 50×50 | 60×60 | 100×100 |
  |---|---|---|---|---|
  | CNN | 0.70 h | 0.78 h | 0.86 h | 1.09 h |
  | Direction-aware GCN | 0.77 h | 2.29 h | 2.97 h | 10.96 h (stopped at 11 h) |

  From the last timestamp of each kernel log (`kaggle/output/chip_size/*/*.log`).

## 6. Kaggle GPU quota (operational, not for the paper)

- At 14:35 UTC on 2026-10-05, with 60×60 and 100×100 done: 24.5 h used of 30 h, so 5.5 h
  are left until the reset on 2026-10-10 at 00:00 UTC.
- 120×120 needs about 11–12.5 GPU-hours (GCN up to 11 h, CNN 0.6–1.2 h), so it waits for
  the reset.
- 120×120 runs on a second account, `deepaksn19` (30 h of fresh quota, same reset date),
  since 2026-10-06 12:17 UTC. It uses about 12.5 GPU-hours, leaving about 17.5 h there,
  enough for the matched-network reference rerun (§10, item 2; about 2 GPU-hours).
- Same platform on both accounts: Kaggle kernels with Tesla T4 GPUs; the GCN and the CNN
  each billed one GPU session.
- Kaggle counts CPU-only kernels toward its 2-session GPU limit (AGENTS.md §5).

## 7. Threats to validity

1. **Single seed** for every three-way comparison and every chip size. The CNN did not
   reproduce at a fixed seed across two Kaggle software environments (67.2% vs. 72.4% at
   epoch 25); the cause is not isolated.
2. **Unconverged baseline** at every size from 30×30 (25 and 40 epochs) to 100×100. Only
   fixed-budget claims hold. At 100×100 the GCN also trained fewer steps (34 vs. 40 epochs,
   11 h limit).
3. **Above 30×30 the input resolution differs** (resampled 30×30 vs. native). No control
   separates resolution from encoder, such as a CNN at native resolution or a GCN on the
   resampled grid.
4. **The direction-aware GCN differs from the baseline in readout, width and embedding size
   at once.** No control such as "Table I convolutions + global max pooling" exists.
5. **Learning-rate schedule asymmetry.** Only methods that exceed 99% decay.
6. **The paper's job distribution is harder than the authors' logged run's.** Our baseline is
   not comparable with their published learning curve.
7. **Healthy chips only.** No faulty or worn chips; the health channel is a constant mask.
8. **Every size is trained from scratch** with the same step budget. Elfar et al. use
   transfer learning across sizes (Fig. 3(b)), so we do not compare against their
   large-chip protocol. 50 and 100 have no published counterpart; 180 is not run.
9. **The 11 h limit binds from 100×100 up.** It cut the 100×100 GCN to 34 epochs, and at
   120×120 only 20–23 epochs fit (F5). Options for 120×120: (a) the same budget for both,
   `EPOCHS=20`, accepting that the GCN may not converge (at 100×100 it was at 91.8% after
   20 epochs); (b) `EPOCHS=40` as at the other sizes, with the GCN stopped by the limit,
   as at 100×100; (c) a resumed second kernel for the GCN, which needs code for resuming a
   run and about 20–23 GPU-hours. The time limit is flagged in `summary.json` and
   `summary.csv`.
10. **Evaluation sets.** Per-epoch metrics come from a fixed 500-job set that also drives
    model selection and learning-rate decay; held-out jobs are not guaranteed disjoint from
    training jobs.

## 8. Wording rules (AGENTS.md §7)

- Write "GCN", never "GNN", in reader-facing text. The result CSVs still carry internal labels:

  | Label in the files | Name in the paper |
  |---|---|
  | `CNN-PPO` | CNN–PPO baseline (our implementation of [elfar2023]) |
  | `GNN-maxpool-PPO`, `GCN-maxpool-PPO` | GCN + global max pooling |
  | `GNN-dirGCN-max-PPO`, `DirGCN-maxpool-PPO` | direction-aware GCN + global max pooling |
  | `GNN-GCN-role-PPO` (16×16 only) | not reported; drop the row |

- Never say our baseline "reproduces" or "replicates" anything. Write "our implementation of
  the CNN–PPO baseline of Elfar et al. [elfar2023]" and "the authors' original
  implementation [elfarcode]". The legend of `results/reference_test/learning_curves.png`
  ("reimplementation") must be relabelled for the paper, and the prose in `docs/RESULTS.md`
  and `README.md` must not be copied.
- Write "Dept. of Computer Science and Engineering". The viva header (line 30) omits
  "Dept. of".
- Label every result as single-seed, including figure captions, and state the
  unconverged-baseline caveat wherever success rates are compared.
- Write "invalid actions per decision" (0.88), not "88% invalid actions".

## 9. Checklist for updating `report/main.tex`

Line numbers refer to the current `report/main.tex`.

- [ ] **Abstract (94–105):** add the main findings (F1, F2) with the single-seed qualifier.
- [ ] **Table 3, configurations and status (569–581):** add the 40-epoch 30×30 run, the
      chip-size study (30–120; GCN + max pooling excluded) and the reference test.
- [ ] **Table 4, parameters (583–611):** add entropy 0.01, clip 0.2, value clip 0.2, gradient
      norm 0.5, the job sampler, the floor health quantization, the evaluation seeds,
      timeout truncation and the 40-epoch budget.
- [ ] **Table 5, platform (646–659):** replace the A100 with the Kaggle T4 and the actual
      Python versions.
- [ ] **Table 7 and text (724–757):** add or replace with the 40-epoch numbers (F2). Rewrite
      the sentence that blames the 25-epoch budget (751–755): the CNN still has not converged
      at 40 epochs. Note that the curve fell from 73.8% to 67.2% in the last two epochs of
      the 25-epoch run.
- [ ] **Figure 7 (740–743):** regenerate from the 40-epoch runs (x-axis to 655 k steps); add
      "single seed" to Figures 6 and 7.
- [ ] **New subsection: validation against the authors' original code** (F4, with its
      caveats and the suggested wording).
- [ ] **New result: chip-size study** (F2 table, F5). `results/chip_size/summary.md` has
      30×30 to 100×100; 120×120 is pending. Mark the 100×100 GCN as "34 epochs (11 h
      limit)".
- [ ] **Method (443–477):**
  - [ ] state the equivalence of the direction-aware GCN with a 3×3 zero-padded CNN with
        global max pooling (F3);
  - [ ] state the D4 invariance for any permutation-invariant readout (F1), not only the
        left–right mirror (703–707);
  - [ ] state the 3-hop receptive field.
- [ ] **Line 240:** H = min(⌊2^b D⌋, 2^b − 1).
- [ ] **Line 257:** the implemented reward is the asymmetric form of the reference code.
- [ ] **Lines 320–322:** Table I prints stride 3; explain the use of stride 1.
- [ ] **Lines 146–149, 224–226, 501–503:** "differ only in the encoder" holds at 16×16 and
      30×30. Above that, the observation resolution differs too.
- [ ] **Figure 1 caption (290–294):** the example is a selected job on a chip with 10% faults;
      the model was trained on healthy chips.
- [ ] **Literature survey (213–218):** add Grid-to-Graph, Zambaldi et al. and Directional
      Graph Networks. Mirhoseini et al. used an edge-based graph network, not a GCN.
- [ ] **Conclusion and future work (760–780):**
  - [ ] the success-rate claim needs the budget qualifier;
  - [ ] the 60×60 and "generalization across sizes" plans are partly done: keep from-scratch
        training per size separate from zero-shot transfer, which has not been run.
- [ ] **Viva (`report/viva/viva_questions.tex`):**
  - [ ] lines 166, 177–178, 229 and 238 are out of date (A100, "16×16 only", "still rising");
  - [ ] lines 132–133 and 216–217 use "GNN";
  - [ ] line 184 ("mean and sum pooling fail in the same way") has no training run behind
        it; only the invariance argument and a random-weight check support it.
- [ ] **Do not cite:**
  - [ ] the CPU timing in `docs/GNN_METHODOLOGY.md` (about 40 ms per batch-32 pass): it has
        no primary source and could not be reproduced;
  - [ ] the authors'-agent check (§4.1) until it is re-run. Their model is a Git LFS object
        (118.95 MB), so fetch it with `git lfs pull` first.

## 10. Open items, in priority order

1. Multi-seed runs (at least 3, ideally 5) for the 30×30 comparison, then for the chip
   sizes.
2. Reference-test rerun with the network and learning-rate rule of `1667016` (F4,
   about 2 GPU-hours).
3. Controls for the confounds of F2: Table I convolutions + global max pooling (readout); a
   GCN on the resampled 30×30 grid or a CNN at native resolution (resolution).
4. A paper-settings vs. 0825a-settings ablation (droplet sizes, k_max basis, collision marks)
   to explain the baseline's slow learning.
5. Committed tests for the D4 invariance (max, mean, sum) and the CNN equivalence, so the
   theory claims can be checked, and one committed script for the audit probes of F1–F3
   (action counts, edge contact, receptive-field probe, router moves) on the held-out jobs.
6. Rerun the authors'-agent check (`scripts/evaluate_reference_model.py`) and commit its
   output.
7. 120×120 after the quota reset; choose among the budget options of §7, item 9.
8. Faulty or worn chips; zero-shot transfer of the GCN across chip sizes.
