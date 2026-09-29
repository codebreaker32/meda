# Methodology in plain words

This file explains what the project does and why, without assuming a
background in reinforcement learning or graph neural networks. The
detailed, code-level version is [docs/GNN_METHODOLOGY.md](docs/GNN_METHODOLOGY.md);
the numbers are in [results/](results/) and the report in [report/](report/).

---

## 1. The problem

A MEDA biochip is a grid of tiny electrodes called **micro-electrode cells
(MCs)**. A droplet sits on a small rectangle of MCs. To move the droplet one
step, the chip switches on the MCs just in front of it and the droplet is
pulled forward.

```
   A 12 x 8 chip                         D = droplet (covers 2x2 MCs)
   . . . . . . . . . . . .               G = goal
   . . . . . . . . . . G G               # = worn-out MCs
   . . . . . . # # . . G G               . = healthy MCs
   . . . . . . # # . . . .
   . D D . . . # # . . . .
   . D D . . . . . . . . .
   . . . . . . . . . . . .
   . . . . . . . . . . . .
```

**The catch: MCs wear out.** Every time an MC is switched on, it gets a
little weaker. A droplet whose front edge is over weak MCs may move slower
than planned, or not at all. The chip can *sense* how healthy each MC is
(a coarse 2-bit health sensor), so a good router looks at those health values and
steers around worn-out areas.

```
   health of one MC over time
   100% |****
        |    *****
        |         *******
        |                ***********
     0% +--------------------------------> number of times switched on
```

**Routing job:** move a droplet from start to goal, inside an allowed zone,
in as few control cycles as possible, without getting stuck.

---

## 2. How the base paper solves it (our baseline)

The base paper (Elfar et al., IEEE TCAD 2023) uses **reinforcement learning
(RL)**: an agent learns by trial and error. It tries moves in a simulator,
gets rewarded for good ones, and slowly learns a policy.

### 2.1 The learning loop

```
        +--------------------------------------------------+
        |                                                  |
        v                                                  |
  +-------------+   what the chip   +----------------+     |
  |   MEDA      |   looks like now  |     AGENT      |     |
  |  simulator  | ----------------> | (encoder + PPO)|     |
  |             |                   |                |     |
  | - wear-out  |   one of 8 moves  |                |     |
  | - sensing   | <---------------- |                |     |
  | - random    |                   +----------------+     |
  |   movement  |                                          |
  +-------------+ ---- reward (good move? reached goal?) --+
```

At every step:

1. The simulator shows the agent the current chip state.
2. The agent picks one of **8 moves**:
   ```
        NW  N  NE
          \ | /
       W -- D -- E
          / | \
        SW  S  SE
   ```
3. The simulator moves the droplet. Whether the move works depends on the
   health of the MCs in front of it, so the outcome is random.
4. The agent gets a **reward**:
   - a small plus for getting closer to the goal;
   - a minus for moving away;
   - +100 for reaching the goal;
   - −1 for an impossible move, such as into the wall.

After millions of such steps, the agent has learned which move to make in
which situation.

### 2.2 What the agent sees

The chip state is given to the agent as **3 layers stacked like a picture**:

```
   layer 1: health        layer 2: droplet       layer 3: goal
   1 1 1 1 1 1            0 0 0 0 0 0            0 0 0 0 1 1
   1 1 .2 .2 1 1          0 0 0 0 0 0            0 0 0 0 1 1
   1 1 .2 .2 1 1          1 1 0 0 0 0            0 0 0 0 0 0
   1 1 1 1 1 1            1 1 0 0 0 0            0 0 0 0 0 0
```

### 2.3 The two parts of the agent

The agent has two jobs, done by two different parts. Keep them apart:

| Part | Job | In the baseline |
|---|---|---|
| **Encoder** | turns the 3-layer picture into a list of numbers that summarise the situation | **CNN** (convolutional neural network), as used for images |
| **Learner** | decides the move from those numbers, and improves from the rewards | **PPO** (Proximal Policy Optimization), with an *actor* (chooses the move) and a *critic* (estimates how good the situation is) |

```
   BASELINE (CNN-PPO)

   3-layer picture --> [ CNN ] --> [ flatten ] --> [ PPO ] --> move (1 of 8)
                        image       long list       actor
                        filters     of numbers      + critic
```

We re-built this baseline from the paper and checked it. The authors' own
trained agent, run in our simulator, routes 100% of 300 jobs in about 10
cycles, as in their log.

---

## 3. Our idea: treat the chip as a graph

A chip is not really a picture. It is a set of MCs, each with a few
neighbours. That is exactly what a **graph** is: dots (**nodes**) joined by
lines (**edges**).

**Our approach replaces only the encoder.** The CNN becomes a GNN. PPO, the
simulator, the rewards and the 8 moves all stay exactly the same, so any
difference in results comes from the encoder.

```
   BASELINE:  3-layer picture --> [ CNN ] --> [ flatten ]     --> [ PPO ] --> move
   OURS:      3-layer picture --> [ graph ] --> [ GNN ] --> [ max pool ] --> [ PPO ] --> move
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  the only part that changes
```

### Words used in this file

- **GNN (graph neural network):** the whole *family* of networks that work
  on graphs. Each node updates itself using messages from its neighbours.
- **GCN (graph convolutional network):** one *particular* GNN, with a
  specific rule for combining messages. It is what the methodology asks for.
- **Direction-aware GCN:** another GNN, the same as a GCN except that each
  of the 8 directions gets its own weights (Section 6).

So "GNN vs GCN" is not a contest: a GCN *is* a GNN, just as a sparrow is a
bird.

---

## 4. The pipeline, step by step

### Step 1: build the graph

- Every MC becomes one **node**, numbered row by row: `node = y*W + x`.
- Every MC is joined to its **8 neighbours**: up, down, left, right and the
  4 diagonals. These match the 8 moves.
- Edges go both ways, so messages can flow in either direction.
- Edges carry no extra information; they only say "these two are
  neighbours".

```
   5 x 4 chip as a graph (numbers = node index)

   15 --- 16 --- 17 --- 18 --- 19
    | \  / | \  / | \  / | \  / |
    |  \/  |  \/  |  \/  |  \/  |
    |  /\  |  /\  |  /\  |  /\  |
    | /  \ | /  \ | /  \ | /  \ |
   10 --- 11 --- 12 --- 13 --- 14
    | \  / | \  / | \  / | \  / |
    |  \/  |  \/  |  \/  |  \/  |
    |  /\  |  /\  |  /\  |  /\  |
    | /  \ | /  \ | /  \ | /  \ |
    5 ---  6 ---  7 ---  8 ---  9
    | \  / | \  / | \  / | \  / |
    |  \/  |  \/  |  \/  |  \/  |
    |  /\  |  /\  |  /\  |  /\  |
    | /  \ | /  \ | /  \ | /  \ |
    0 ---  1 ---  2 ---  3 ---  4

   corner node: 3 neighbours    edge node: 5    inside node: 8
   a 30 x 30 chip has 900 nodes and 6,844 one-way edges
```

### Step 2: give each node its features

Each node gets the **same 3 numbers** the CNN saw at that MC, and nothing
more:

```
   node i  -->  [ health_i , droplet_i , goal_i ]

   e.g.  node 12 = [ 0.25 , 0 , 0 ]     worn-out MC, no droplet, not goal
         node  6 = [ 1.00 , 1 , 0 ]     healthy MC under the droplet
```

A worn-out MC is **not removed** from the graph. It stays a node, carrying
its low health value, and the network learns to avoid it.

### Step 3: message passing (the GNN layers)

In each layer, every node collects messages from its neighbours and updates
itself. After 3 layers, each node "knows" about everything up to 3 steps
away.

```
   layer 0          layer 1              layer 2              layer 3
   (own info)       (+ 1 step away)      (+ 2 steps away)     (+ 3 steps away)

   . . . . . . .    . . . . . . .        . . . . . . .        x x x x x x x
   . . . . . . .    . . . . . . .        . x x x x x .        x x x x x x x
   . . . . . . .    . . x x x . .        . x x x x x .        x x x x x x x
   . . . o . . .    . . x o x . .        . x x o x x .        x x x o x x x
   . . . . . . .    . . x x x . .        . x x x x x .        x x x x x x x
   . . . . . . .    . . . . . . .        . x x x x x .        x x x x x x x
   . . . . . . .    . . . . . . .        . . . . . . .        x x x x x x x

   o = a node        x = MCs whose information has reached it
```

A simulated example of what one node receives in one layer:

```
   node 7 (worn MC) sends: "I am worn out"   ----\
   node 2 (healthy) sends: "I am fine"       -----+--> node 6 combines these
   node 12 (goal!) sends:  "goal is here"    ----/     and updates itself
   ... (8 neighbours in total)
```

### Step 4: global max pooling

After message passing, each node has a list of 64 numbers (its
*embedding*). PPO needs **one** list for the whole chip, so for each of
the 64 positions we take the **largest value over all nodes**:

```
              pos 1  pos 2  pos 3  ...  pos 64
   node 0  [  0.1    0.0    0.7   ...   0.2 ]
   node 1  [  0.9    0.3    0.1   ...   0.0 ]
   node 2  [  0.2    0.8    0.4   ...   0.5 ]
    ...
   -----------------------------------------
   max     [  0.9    0.8    0.7   ...   0.5 ]   <-- the chip summary given to PPO
```

This is simple and works for any chip size. Its weakness is that it
forgets *which* node gave each maximum. That turns out to matter
(Section 6).

### Step 5: PPO, unchanged

PPO takes that summary list and outputs:

- **actor:** the probability of each of the 8 moves;
- **critic:** how good the current situation is.

Training works exactly as in the baseline.

---

## 5. How we test it fairly

```
   +--------------------+        same simulator, rewards, moves,
   | CNN-PPO (baseline) |---+    PPO settings, training budget
   +--------------------+   |
                            +--> train --> evaluate on the SAME 500
   +--------------------+   |               held-out routing jobs
   | GNN-PPO (ours)     |---+               (never seen in training)
   +--------------------+                          |
                                                   v
                             success rate, routing cycles, wrong-move rate,
                             steps to learn, training time, decision time
```

- **Same settings:** the GNN config file *inherits* every setting from the
  CNN one and changes only the encoder. A test checks this.
- **Ablations:** each changes exactly **one** thing, so we know what
  caused a difference:

| Experiment | What changes | Question it answers |
|---|---|---|
| CNN-PPO | (baseline) | how good is the paper's method? |
| GCN + max pooling | encoder = GCN | does the proposed method work? |
| Direction-aware GCN + max pooling | layer only | is the GCN layer the problem? |
| GCN + role-aware readout | pooling only | is max pooling the problem? |

---

## 6. What we found (preliminary: 16x16 chip, 1 seed)

| Method | Success | Avg. cycles | Wrong moves | Parameters |
|---|---|---|---|---|
| CNN-PPO (baseline) | 96.2% | 5.94 | 12% | 2.15 M |
| GCN + max pooling | 4.0% | 25.06 | 88% | 8.6 k |
| **Direction-aware GCN + max pooling** | **99.4%** | **4.89** | **0.4%** | 76 k |
| GCN + role-aware readout | 5.0% | 24.83 | 89% | 10 k |

### Why the plain GCN failed: it cannot tell left from right

A GCN adds up its neighbours' messages **with the same weights**, whatever
direction they come from. Now look at a job and its mirror image:

```
   job A: goal is EAST             job B: goal is WEST (mirror of A)

   . . . . . . . .                 . . . . . . . .
   . D D . . . G G                 G G . . . D D .
   . D D . . . G G                 G G . . . D D .
   . . . . . . . .                 . . . . . . . .

   correct move: E                 correct move: W
```

- **Same neighbourhoods:** every node in job B sees exactly the same
  neighbours as its mirror node in job A, only on the other side. A GCN
  ignores sides, so the mirrored nodes get identical embeddings.
- **Same summary:** max pooling takes the maximum over *all* nodes, so
  both jobs give the **same chip summary**.
- **Same decision:** PPO sees the same numbers for both jobs, so it picks
  the same move, but the right answers are opposite.

It is like asking someone for directions who can see the map but has no
compass. A test in the code (`tests/test_graph_readout.py`) proves this.

### The fix: give each direction its own weights

The **direction-aware GCN** (a relational GCN) uses **separate weights
for each of the 8 directions**. A message from the east neighbour is
treated differently from one from the west:

```
   plain GCN:                        direction-aware GCN:

        W   W   W                        W_NW  W_N  W_NE
          \ | /                              \  |  /
     W --  node  -- W                  W_W -- node -- W_E
          / | \                              /  |  \
        W   W   W                        W_SW  W_S  W_SE

   one weight for all                one weight per direction
   -> no sense of direction          -> knows where things are
```

- **Only the layer changed:** the graph and max pooling stay exactly the
  same.
- **Result:** success went from **4% to 99.4%**, better than the CNN,
  with about 28 times fewer parameters and learning in about a third of
  the steps.

The role-aware readout, which changes the pooling instead, did *not* help
(5%). **Max pooling was not the problem; the GCN layer was.**

### Honest caveat

These results come from one seed on a small chip. They will be repeated
with 5 seeds on 30x30 chips, as in the paper, before we draw conclusions.

---

## 7. Where each piece lives in the code

```
   observation --> graph_builder.py --> gnn.py --> graph_readout.py --> PPO
                   (Steps 1-2)          (Step 3)   (Step 4)             (Step 5)
```

| Piece | File |
|---|---|
| Simulator (chip, wear-out, moves, rewards) | `src/meda_routing/envs/` |
| CNN encoder (baseline) | `src/meda_routing/agents/cnn.py` |
| Graph construction, 8 neighbours | `src/meda_routing/representations/graph_builder.py` |
| GCN and direction-aware layers | `src/meda_routing/agents/gnn.py` |
| Max pooling and role-aware readout | `src/meda_routing/agents/graph_readout.py` |
| Fair comparison on the same jobs | `src/meda_routing/experiments/method_comparison.py` |
| Experiment settings (one file per row of Section 5) | `configs/training/` |
| Run everything | `scripts/run_gnn_experiment.sh` |
| Run on the GPU server | `scripts/run_on_gpu_server.sh` |
| Results (tables, plots) | `results/` |

To run the small version on a laptop (about 1 hour on 4 CPU cores):

```bash
SIZE=16 SEEDS=1 bash scripts/run_gnn_experiment.sh
```

To run the full version (30x30, 5 seeds) on the GPU server:

```bash
GPU_SERVER=user@192.168.x.x bash scripts/run_on_gpu_server.sh bash scripts/run_gnn_experiment.sh
```
