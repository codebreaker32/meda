"""Kaggle kernel: train the authors' original code (melfar87/MEDA, unmodified) with the
settings of their logged 30x30 run 0825a, for the seeds in SEEDS (one after the other).

kaggle/push_reference.sh fills in SEEDS and pushes it. CPU only: the code needs
tensorflow 1.15 (CUDA 10), which Kaggle's GPUs no longer support; one 2^14-step epoch
takes about 5 min on 4 cores. Outputs: /kaggle/working/original/seed_<s>/progress.csv
(written after every epoch) and the authors' pickle.
"""
import subprocess

SEEDS = __SEEDS__
EPOCHS = 40
BRANCH = "meda/gnn/graph_routing"
REPO = "https://github.com/codebreaker32/meda"
ORIG_REPO = "https://github.com/melfar87/MEDA"
ORIG_COMMIT = "1667016"   # the commit the reference results were checked against

WORK = "/kaggle/working/original"
TMP = "/kaggle/tmp"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"rm -rf {TMP} && mkdir -p {TMP} {WORK}")
sh(f"git clone --depth 1 -b {BRANCH} {REPO} {TMP}/meda")
sh(f"GIT_LFS_SKIP_SMUDGE=1 git clone {ORIG_REPO} {TMP}/orig && git -C {TMP}/orig checkout -q {ORIG_COMMIT}")
sh(f"bash {TMP}/meda/scripts/reference/setup_original_env.sh {TMP}/tf1")
for seed in SEEDS:
    sh(f"{TMP}/tf1/bin/python -u {TMP}/meda/scripts/reference/run_original_0825a.py "
       f"--repo {TMP}/orig --epochs {EPOCHS} --seed {seed} --out {WORK}/seed_{seed}")
    print("done: seed", seed, flush=True)
