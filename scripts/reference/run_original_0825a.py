"""Train the authors' original code (melfar87/MEDA, unmodified) with the settings of
their logged 30x30 run "0825a" (policy/0825a_030x030_E100_NPS64.pickle).

Needs the authors' software stack: Python 3.7, tensorflow 1.15, stable-baselines 2.10.1,
gym 0.18 (see scripts/reference/setup_original_env.sh). Run from anywhere:

    python run_original_0825a.py --repo /path/to/melfar87_MEDA --epochs 40 --seed 0 --out out_dir

The argument set is taken verbatim from the run's own log; only n_epochs, seed and
b_save_model are changed. The per-epoch evaluation (500 jobs, deterministic policy) is
the authors' EvaluatePolicy; it is written to <out>/progress.csv after every epoch, and
the authors' pickle (same format as theirs) to <out>/<name>.pickle at the end.
"""
import argparse
import csv
import os
import pickle
import re
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True, help="clone of github.com/melfar87/MEDA")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=None, help="steps per epoch (default: the log's 2^14)")
p.add_argument("--evals", type=int, default=None, help="evaluation jobs (default: the log's 500)")
p.add_argument("--out", required=True)
a = p.parse_args()

repo = os.path.abspath(a.repo)
out = os.path.abspath(a.out)
os.makedirs(out, exist_ok=True)
with open(os.path.join(repo, "policy", "0825a_030x030_E100_NPS64.pickle"), "rb") as fh:
    args = pickle.load(fh)["args"]          # the run's own argparse.Namespace
args.n_epochs = a.epochs
args.seed = a.seed
args.b_save_model = False
args.s_model_name = "orig0825a"
args.s_suffix = "s%d" % a.seed
if a.steps:
    args.n_total_timesteps = a.steps
if a.evals:
    args.n_evals = a.evals
print("args:", vars(args), flush=True)

# train.py does its imports under `if __name__ == '__main__'`; repeat them here.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args.verbose)[0]
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"The name")
os.chdir(out)                                # train.py writes data/ and log/ relative to cwd
os.makedirs("data", exist_ok=True)
os.makedirs("log", exist_ok=True)
sys.path.insert(0, repo)
import numpy as np
np.set_printoptions(linewidth=np.inf)
np.seterr(all="raise")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import train                                  # the authors' module, unmodified
from my_net import MyCnnPolicy
from envs.meda import MEDAEnv
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.misc_util import set_global_seeds
for name, obj in dict(np=np, plt=plt, tf=tf, MyCnnPolicy=MyCnnPolicy, MEDAEnv=MEDAEnv,
                      make_vec_env=make_vec_env, PPO2=PPO2).items():
    setattr(train, name, obj)
if args.seed >= 0:
    set_global_seeds(args.seed)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
train.plotAgentPerformance = lambda *x, **k: None   # needs tikzplotlib; the pickle has the data
os.system = lambda *x, **k: 0                        # train.py clears the terminal

# Record every epoch as it is printed (the pickle is only written at the end).
LINE = re.compile(r"Exp/Epoc (\d+)-(\d+)\s+(\d+)/(\d+) sec\s+(\S+) rew\s+(\S+) suc\s+(\S+) cyc\s+(\S+)")
progress = open(os.path.join(out, "progress.csv"), "w", newline="")
writer = csv.writer(progress)
writer.writerow(["epoch", "timesteps", "success_rate", "mean_cycles", "mean_score", "lr_base", "epoch_seconds"])


class Tee:
    def __init__(self, stream):
        self.stream, self.buf = stream, ""

    def write(self, s):
        self.stream.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            m = LINE.search(line)
            if m:
                ep = int(m.group(2)) + 1
                writer.writerow([ep, ep * args.n_total_timesteps, float(m.group(6)) / 100,
                                 float(m.group(7)), float(m.group(5)), float(m.group(8)), int(m.group(4))])
                progress.flush()

    def flush(self):
        self.stream.flush()


sys.stdout = Tee(sys.stdout)
t0 = time.time()
train.main(args)
print("total seconds:", round(time.time() - t0), flush=True)
