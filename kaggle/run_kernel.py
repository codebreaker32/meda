"""Kaggle kernel: train one 30x30 configuration and evaluate it on the held-out jobs.

kaggle/push_30x30.sh fills in the three settings below and pushes one kernel per
configuration. Everything the run produces is written to /kaggle/working:
  runs/<name>/seed_<s>/   model.zip, best_model.zip, progress.csv, training curves
  results/                held-out evaluation (tables, logs, figures)
"""
import os
import shutil
import subprocess

CONFIG = "__CONFIG__"    # configs/training/<CONFIG>.yaml
LABEL = "__LABEL__"      # method label in the results tables
SEEDS = __SEEDS__        # training repeats (seeds 0..SEEDS-1); the paper uses 5 [PAPER Sec. V-B]
BRANCH = "meda/gnn/graph_routing"
REPO = "https://github.com/codebreaker32/meda"
EPISODES = 500           # held-out evaluation jobs, same seed for every configuration [PAPER Sec. V-B]

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

name = subprocess.run(
    f"python -c \"import yaml;print(yaml.safe_load(open('{SRC}/configs/training/{CONFIG}.yaml'))['name'])\"",
    shell=True, check=True, capture_output=True, text=True).stdout.strip()

sets = (f"--set repeats={SEEDS} --set output_dir={WORK}/runs --set ppo.device={device} "
        "--set schedule.checkpoint_every=0")   # no per-epoch checkpoints: the CNN model is 0.3 GB
os.chdir(SRC)
sh("meda devices")
sh(f"meda train -c configs/training/{CONFIG}.yaml {sets}")
sh(f"meda compare-methods --method {LABEL} {WORK}/runs/{name} --episodes {EPISODES} "
   f"--device {device} --out {WORK}/results")
shutil.rmtree(SRC, ignore_errors=True)
print("done:", name, flush=True)
