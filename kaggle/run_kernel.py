"""Kaggle kernel: train the configurations in JOBS and evaluate each on the held-out jobs.

kaggle/push_30x30.sh and kaggle/push_sizes.sh fill in the settings below and push
the kernels (Kaggle runs at most two at a time); a kernel trains its
configurations one after the other. Everything is written to /kaggle/working:
  runs/<name>/seed_<s>/   model.zip, best_model.zip, progress.csv, training curves
  results/                held-out evaluation (tables, logs, figures)
"""
import os
import shutil
import subprocess
import time

T0 = time.time()

# (config in configs/training/, method label in the results tables), trained in order
JOBS = [__JOBS__]
SEEDS = __SEEDS__        # training repeats (seeds 0..SEEDS-1); the paper uses 5 [PAPER Sec. V-B]
BRANCH = "meda/gnn/graph_routing"
REPO = "https://github.com/codebreaker32/meda"
EPISODES = 500           # held-out evaluation jobs, same seed for every configuration [PAPER Sec. V-B]
EXTRA_SETS = ""          # extra "--set key=value" overrides (push_sizes.sh: EXTRA=...)
TRAIN_HOURS = 11.0       # all training ends by then (schedule.max_hours); Kaggle stops kernels at 12 h

WORK = "/kaggle/working"
SRC = "/kaggle/tmp/meda"   # outside /kaggle/working, so the code is not part of the output


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"rm -rf {SRC} && git clone --depth 1 -b {BRANCH} {REPO} {SRC}")
sh("pip install -q 'stable-baselines3>=2.2' 'gymnasium>=0.29' 'opencv-python-headless>=4.5' pyyaml")
sh(f"pip install -q --no-deps -e {SRC}")

# Kaggle's P100 is not supported by recent PyTorch builds; fall back to the CPU then.
device = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        x = torch.randn(256, 256, device="cuda")
        (x @ x).sum().item()
        device = "cuda"
except Exception as exc:  # noqa: BLE001 - any CUDA failure means CPU
    print("CUDA not usable, training on the CPU:", exc, flush=True)
print("device:", device, flush=True)

os.chdir(SRC)
sh("meda devices")
sets = (f"--set repeats={SEEDS} --set output_dir={WORK}/runs --set ppo.device={device} "
        "--set schedule.checkpoint_every=0 " + EXTRA_SETS)   # no per-epoch checkpoints: the CNN model is 0.3 GB
for i, (config, label) in enumerate(JOBS):
    name = subprocess.run(
        f"python -c \"import yaml;print(yaml.safe_load(open('configs/training/{config}.yaml'))['name'])\"",
        shell=True, check=True, capture_output=True, text=True).stdout.strip()
    # the hours left are shared equally by the remaining runs (jobs x seeds, one after the other)
    per_seed = (TRAIN_HOURS - (time.time() - T0) / 3600) / ((len(JOBS) - i) * SEEDS)
    sh(f"meda train -c configs/training/{config}.yaml --set schedule.max_hours={per_seed:.3f} {sets}")
    sh(f"meda compare-methods --method {label} {WORK}/runs/{name} --episodes {EPISODES} "
       f"--device {device} --out {WORK}/results/{name}")
    print("done:", name, flush=True)
shutil.rmtree(SRC, ignore_errors=True)
