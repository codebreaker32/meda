# Methodology Overview

This document summarises the project's approach for readers without a
background in reinforcement learning or graph neural networks. The formal,
code-level specification is [docs/GNN_METHODOLOGY.md](docs/GNN_METHODOLOGY.md);
experimental outputs are in [results/](results/) and the mid-semester report
is in [report/](report/).

**Contents**

1. [Problem definition](#1-problem-definition)
2. [Baseline method: CNN–PPO](#2-baseline-method-cnnppo)
3. [Proposed method: GNN–PPO](#3-proposed-method-gnnppo)
4. [Proposed pipeline](#4-proposed-pipeline)
5. [Experimental design](#5-experimental-design)
6. [Preliminary results](#6-preliminary-results)
7. [Implementation map](#7-implementation-map)

---

## 1. Problem definition

A micro-electrode-dot-array (MEDA) biochip consists of a grid of
**micro-electrode cells (MCs)**. A droplet occupies a rectangular block of
MCs and is moved by actuating the MCs adjacent to its leading edge.

```
   12 x 8 chip                           D  droplet (occupies 2 x 2 MCs)
   . . . . . . . . . . . .               G  goal location
   . . . . . . . . . . G G               #  degraded MCs
   . . . . . . # # . . G G               .  healthy MCs
   . . . . . . # # . . . .
   . D D . . . # # . . . .
   . D D . . . . . . . . .
   . . . . . . . . . . . .
   . . . . . . . . . . . .
```

**MC degradation.** Each actuation reduces an MC's actuation force. When
the MCs at the droplet's leading edge are degraded, the droplet may move
more slowly than commanded or fail to move. Each MC reports its condition
through an on-chip health sensor (b = 2 bits), so a routing policy can take
MC health into account and avoid degraded regions.

```
   MC health versus number of actuations   (model: D = tau^(n/c))

   100% |****
        |    *****
        |         *******
        |                ***********
     0% +--------------------------------> number of actuations n
```

**Routing job.** Given a droplet, a start location, a goal location and a
permitted routing zone, move the droplet to the goal in as few control
cycles as possible.

---

## 2. Baseline method: CNN–PPO

The base paper (Elfar et al., IEEE TCAD 2023) formulates droplet routing as
a **reinforcement-learning (RL)** problem. An agent interacts with a
simulator, receives a reward after each action, and learns a policy that
maximises the cumulative reward.

### 2.1 Agent–environment loop

```
        +--------------------------------------------------+
        |                                                  |
        v                                                  |
  +-------------+   observation     +----------------+     |
  |    MEDA     |   (chip state)    |     Agent      |     |
  |  simulator  | ----------------> | (encoder + PPO)|     |
  |             |                   |                |     |
  | degradation |   action          |                |     |
  | sensing     | <---------------- |                |     |
  | stochastic  |   (1 of 8)        +----------------+     |
  | movement    |                                          |
  +-------------+ ------ reward, episode termination ------+
```

At each control cycle:

1. The simulator provides the current observation of the chip.
2. The agent selects one of **eight actions**:
   ```
        NW  N  NE
          \ | /
       W -- D -- E
          / | \
        SW  S  SE
   ```
3. The simulator applies the action. Whether the droplet moves depends
   stochastically on the health of the MCs at its leading edge.
4. The agent receives a reward (coefficients from the authors' reference
   code):
   - +0.5 per MC of progress towards the goal;
   - a penalty for a move that makes no progress (0.8 per MC of distance
     lost, plus 1);
   - +100 on reaching the goal;
   - −1 for an invalid action, such as a move into the chip boundary.

Training repeats this cycle for several hundred thousand environment steps
(327,680 in the 16×16 study).

### 2.2 Observation

The chip state is presented to the agent as a **three-channel grid**, one
value per MC in each channel:

```
   channel 1: health          channel 2: droplet      channel 3: goal
   1  1  1  1  1  1           0 0 0 0 0 0             0 0 0 0 1 1
   1  1 .5 .5  1  1           0 0 0 0 0 0             0 0 0 0 1 1
   1  1 .5 .5  1  1           1 1 0 0 0 0             0 0 0 0 0 0
   1  1  1  1  1  1           1 1 0 0 0 0             0 0 0 0 0 0
```

Health is normalised to [0, 1] and set to 0 outside the routing zone.

### 2.3 Agent architecture

The agent comprises two components with distinct roles:

| Component | Function | Baseline implementation |
|---|---|---|
| **State encoder** | Maps the observation to a fixed-length feature vector | **CNN**: three 3×3 convolutional layers followed by a fully connected layer (Table I of the paper) |
| **Learning algorithm** | Selects actions from the feature vector and updates the policy from the rewards | **PPO** (Proximal Policy Optimization): an *actor* that outputs action probabilities and a *critic* that estimates the value of the current state |

```
   Baseline (CNN–PPO)

   observation --> [ image convolution x3 ] --> [ flatten + FC ] --> [ PPO ] --> action (1 of 8)
                     3x3 kernels, 3 layers        feature vector      actor + critic
                   \_______________ CNN encoder _______________/
```

**Validation of the reimplementation.** The authors' own trained 30×30
agent, evaluated in our simulator, routes 100% of 300 random jobs in 10.0
cycles on average, consistent with their training log (99.8–100%,
approximately 10.5 cycles).

---

## 3. Proposed method: GNN–PPO

### 3.1 Motivation

A MEDA chip is a set of MCs with fixed spatial adjacency, which maps
directly onto a **graph**: MCs become nodes and adjacent MCs are joined by
edges. A graph encoder applies the same weights at every node, so its
parameter count is independent of the chip size, and it operates on the
native MC grid without resizing the observation.

### 3.2 Scope of the change

**Only the state encoder is replaced.** The simulator, reward, action
space, PPO and its hyperparameters, training budget and evaluation jobs
are unchanged, so any difference in performance is attributable to the
encoder.

```
   Baseline:  observation --> [ image convolution x3 ] ----------------> [ flatten + FC ] --> [ PPO ] --> action

   Proposed:  observation --> [ graph ] --> [ graph convolution x3 ] --> [ max pooling  ] --> [ PPO ] --> action
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              replaced component (state encoder)
```

In both pipelines, the convolutional layers reside in the state encoder:
image convolutions in the CNN and graph convolutions in the GNN
(Section 4.4).

### 3.3 Terminology

- **Graph neural network (GNN):** the general class of neural networks
  that operate on graphs. Each node updates its representation from
  messages sent by its neighbours.
- **Graph convolutional network (GCN):** a specific GNN (Kipf and Welling,
  2017) defined by a particular message-aggregation rule. It is the GNN
  specified by the project methodology.
- **Direction-aware GCN:** a relational GCN (Schlichtkrull et al., 2018)
  that assigns separate weights to each of the eight neighbour directions
  (Section 6.3).

A GCN is therefore one instance of a GNN; the two terms are not
alternatives.

---

## 4. Proposed pipeline

```
   observation (3 channels, W x H)
        |
        v
   [4.1] graph construction        one node per MC, 8-neighbour edges
        |
        v
   [4.2] node features             [health, droplet, goal] per node     X: N x 3
        |
        v
   [4.3] graph convolution x3      message passing between neighbours   Z: N x 64
        |
        v
   [4.5] global max pooling        maximum over all nodes               g: 64
        |
        v
   [4.6] PPO actor and critic      action probabilities, state value
```

N is the number of MCs (nodes). Section 4.4 relates the graph convolution
to the image convolution of the baseline CNN, and Section 4.7 works through
one convolution layer numerically.

### 4.1 Graph construction

- Each MC is represented by one **node**, indexed in row-major order:
  `node = y * W + x`.
- Each MC is connected to its **eight neighbours**: horizontal, vertical
  and diagonal. These correspond to the eight actions.
- Edges are stored in both directions, so information can propagate either
  way.
- Edges carry no attributes; they encode adjacency only.

```
   5 x 4 chip represented as a graph (labels are node indices)

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

   Degree: corner nodes 3, boundary nodes 5, interior nodes 8.
   A 30 x 30 chip yields 900 nodes and 6,844 directed edges.
```

### 4.2 Node features

Each node carries **exactly the three values** that the CNN observes at the
corresponding MC:

```
   node i  -->  [ health_i , droplet_i , goal_i ]

   Example:  node 12 = [ 0.25 , 0 , 0 ]    degraded MC, unoccupied, not a goal MC
             node  6 = [ 1.00 , 1 , 0 ]    healthy MC occupied by the droplet
```

Degraded MCs are **retained** as nodes. Their low health value is part of
the input, and the policy learns to avoid them.

### 4.3 Graph convolution (message passing)

In each GNN layer, every node aggregates messages from its neighbours and
updates its representation (its *embedding*). With three layers, each
node's embedding depends on all MCs within three steps, i.e. a 7×7 block.

```
   layer 0          layer 1              layer 2              layer 3
   (node only)      (1-hop)              (2-hop)              (3-hop)

   . . . . . . .    . . . . . . .        . . . . . . .        x x x x x x x
   . . . . . . .    . . . . . . .        . x x x x x .        x x x x x x x
   . . . . . . .    . . x x x . .        . x x x x x .        x x x x x x x
   . . . o . . .    . . x o x . .        . x x o x x .        x x x o x x x
   . . . . . . .    . . x x x . .        . x x x x x .        x x x x x x x
   . . . . . . .    . . . . . . .        . x x x x x .        x x x x x x x
   . . . . . . .    . . . . . . .        . . . . . . .        x x x x x x x

   o  target node        x  MCs within its receptive field
```

Example of one aggregation step at node 6 of the graph in Section 4.1:

```
   node 7   degraded MC    [0.25, 0, 0]  ----\
   node 2   healthy MC     [1.00, 0, 0]  -----+-->  node 6 aggregates the
   node 5   droplet MC     [1.00, 1, 0]  ----/      messages and updates
   ...      (8 neighbours in total)                 its embedding
```

The numerical values of this aggregation are given in Section 4.7.

### 4.4 Relation to image convolution

Section 4.3 is the convolutional stage of the proposed encoder: each GNN
layer performs one **graph convolution**, the generalisation to graphs of
the image convolution used by the CNN. Graph construction (Sections
4.1–4.2) and pooling (Section 4.5) contain no convolution.

| Encoder | Convolutional layers | Implementation |
|---|---|---|
| Baseline CNN | 3 image convolutions, 3×3 kernels | `agents/cnn.py`, `nn.Conv2d` (line 49) |
| GCN | 3 graph convolutions | `agents/gnn.py`, `GCNLayer` (line 58) |
| Direction-aware GCN | 3 graph convolutions with direction-specific weights | `agents/gnn.py`, `DirectionalLayer` (line 70) |

On the MC grid, all three operate on the same 3×3 neighbourhood: the MC
itself and its eight neighbours. They differ in how many independent weight
matrices they apply to these nine positions:

```
   CNN 3x3 kernel          GCN                        Direction-aware GCN
   w1 w2 w3                a  a  a                    W_NW  W_N  W_NE
   w4 w5 w6                a  a  a                    W_W   W_0  W_E
   w7 w8 w9                a  a  a                    W_SW  W_S  W_SE

   9 independent           1 shared weight matrix,    9 independent weight
   weights                 scaled by fixed degree     matrices (node itself
                           normalisation              and 8 directions)
```

- **GCN:** a single weight matrix is shared by all nine positions, so the
  operation is a normalised neighbourhood average. It is isotropic and
  cannot distinguish one direction from another (Section 6.2).
- **Direction-aware GCN:** on a regular grid, this layer is equivalent to
  a 3×3 convolution with zero padding, as used by the baseline CNN. It
  differs from the baseline in the rest of the encoder: there is no
  flatten and fully connected layer, the readout is max pooling, and the
  parameter count does not depend on the chip size. The absence of the
  fully connected layer accounts for most of the difference in parameter
  count (76 k versus 2.15 M on 16×16).

### 4.5 Global max pooling

After message passing, each node has a 64-dimensional embedding. PPO
requires a single vector for the whole chip, so each dimension is reduced
to its **maximum over all nodes**:

```
                 dim 1  dim 2  dim 3  ...  dim 64
   node 0     [  0.1    0.0    0.7    ...   0.2  ]
   node 1     [  0.9    0.3    0.1    ...   0.0  ]
   node 2     [  0.2    0.8    0.4    ...   0.5  ]
    ...
   ------------------------------------------------
   max        [  0.9    0.8    0.7    ...   0.5  ]   <-- graph embedding passed to PPO
```

The operation is independent of node ordering and produces a vector of
fixed length for any chip size. It does not retain which node produced each
maximum; the consequences are examined in Section 6.2.

### 4.6 Policy and value heads

PPO is unchanged. Linear heads on the graph embedding produce:

- **Actor:** a probability distribution over the eight actions.
- **Critic:** an estimate of the value of the current state.

### 4.7 Worked example: one convolution layer

This example applies one layer of each graph encoder to the 5×4 chip of
Section 4.1, followed by global max pooling. The weights are set by hand for
illustration; in training they are learned. All values were computed with
the project's layer implementations (`GCNLayer` and `DirectionalLayer` in
`agents/gnn.py`), with one output dimension and zero bias.

**Chip state.** All MCs lie inside the routing zone.

```
   y = 3    15 .    16 .    17 .    18 G    19 G        D  droplet MC    features [1.00, 1, 0]
   y = 2    10 .    11 .    12 #    13 G    14 G        G  goal MC       features [1.00, 0, 1]
   y = 1     5 D     6 D     7 #     8 .     9 .        #  degraded MC   features [0.25, 0, 0]
   y = 0     0 D     1 D     2 .     3 .     4 .        .  healthy MC    features [1.00, 0, 0]
           x = 0   x = 1   x = 2   x = 3   x = 4
```

**Step 1: message computation.** Each node's feature vector
x = [health, droplet, goal] is multiplied by the layer's weight vector,
here w = [1.0, 0.5, 2.0]:

```
   m_j = w . x_j = 1.0 * health + 0.5 * droplet + 2.0 * goal

   droplet MC    [1.00, 1, 0]   ->   1.00 + 0.50          = 1.50
   goal MC       [1.00, 0, 1]   ->   1.00 + 2.00          = 3.00
   degraded MC   [0.25, 0, 0]   ->   0.25                 = 0.25
   healthy MC    [1.00, 0, 0]   ->   1.00                 = 1.00
```

**Step 2: GCN aggregation at node 6.** Node 6 combines its own message and
those of its eight neighbours. Each message is scaled by the fixed
coefficient 1/√(d_j · d_6), where d is the node degree including the
node itself (d_6 = 9):

| Node j | Position | MC type | m_j | d_j | 1/√(d_j · d_6) | Contribution |
|---|---|---|---|---|---|---|
| 10 | NW | healthy | 1.00 | 6 | 0.136 | 0.136 |
| 11 | N | healthy | 1.00 | 9 | 0.111 | 0.111 |
| 12 | NE | degraded | 0.25 | 9 | 0.111 | 0.028 |
| 5 | W | droplet | 1.50 | 6 | 0.136 | 0.204 |
| 6 | node itself | droplet | 1.50 | 9 | 0.111 | 0.167 |
| 7 | E | degraded | 0.25 | 9 | 0.111 | 0.028 |
| 0 | SW | droplet | 1.50 | 4 | 0.167 | 0.250 |
| 1 | S | droplet | 1.50 | 6 | 0.136 | 0.204 |
| 2 | SE | healthy | 1.00 | 6 | 0.136 | 0.136 |
| | | | | | **Sum** | **1.264** |

The layer output is ReLU(1.264) = **1.264**. The coefficients depend only
on the node degrees, not on the direction of the neighbour.

**Step 3: direction-aware aggregation at node 6.** The direction-aware
layer applies a separate weight to each of the nine positions. For
illustration, the weight is +1 for the three eastern positions, −1 for the
three western positions and 0 elsewhere. (Formally, each direction's weight
matrix is W_r = c_r · w, so the calculation reduces to one scalar c_r per
position.) Arranged as a 3×3 patch around node 6, with north at the top:

```
   messages m_j                 weights c_r                 products c_r * m_j

   1.00   1.00   0.25           -1    0   +1                -1.00   0.00   +0.25
   1.50   1.50   0.25     x     -1    0   +1       =        -1.50   0.00   +0.25
   1.50   1.50   1.00           -1    0   +1                -1.50   0.00   +1.00

                                              sum = 1.50 - 4.00 = -2.50
```

This multiply-and-sum over a 3×3 patch is exactly the operation of an image
convolution with a 3×3 kernel. The layer output is ReLU(−2.50) = **0.00**.
The first layer of the baseline CNN performs the same operation directly on
the observation, with 3×3 kernels spanning its three channels.

**Step 4: the mirror-image job.** Reflecting the job left to right moves
the droplet to x = 3–4 and the goal to x = 0–1. The node that corresponds
to node 6 is node 8.

| | GCN | Direction-aware GCN (before ReLU) |
|---|---|---|
| Job A (original), node 6 | 1.264 | −2.50 |
| Job B (mirror image), node 8 | 1.264 | +2.50 |

The GCN result is unchanged, because node 8 has the same neighbours with the
same degrees, only reflected. In the direction-aware layer, the eastern and
western columns exchange places and the result changes sign.

**Step 5: global max pooling.** Applying each layer to all 20 nodes and
taking the maximum gives the graph embedding passed to PPO:

```
   GCN output, job A                            GCN output, job B (mirror image)
   y=3   0.825  0.874  1.412  2.221  2.475      y=3   2.475  2.221  1.412  0.874  0.825
   y=2   1.128  1.112  1.458  2.089  2.323      y=2   2.323  2.089  1.458  1.112  1.128
   y=1   1.313  1.264  1.254  1.483  1.582      y=1   1.582  1.483  1.254  1.264  1.313
   y=0   1.237  1.211  0.958  0.874  0.825      y=0   0.825  0.874  0.958  1.211  1.237
           x=0    x=1    x=2    x=3    x=4              x=0    x=1    x=2    x=3    x=4
                       max pooling = 2.475                          max pooling = 2.475

   Direction-aware output, job A                Direction-aware output, job B (mirror image)
   y=3    2.00   0.00   4.00   4.75   0.00      y=3    6.00   0.00   0.00   0.75   0.00
   y=2    3.50   0.00   3.50   5.50   0.00      y=2    7.00   0.00   0.00   2.00   0.00
   y=1    4.00   0.00   1.00   3.50   0.00      y=1    5.00   0.00   0.00   2.50   0.00
   y=0    3.00   0.00   0.00   0.75   0.00      y=0    2.00   0.00   1.00   1.75   0.00
           x=0    x=1    x=2    x=3    x=4              x=0    x=1    x=2    x=3    x=4
                        max pooling = 5.50                           max pooling = 7.00
```

| Encoder | Pooled value, job A | Pooled value, job B | Can the policy distinguish A from B? |
|---|---|---|---|
| GCN | 2.475 | 2.475 | No: the output maps are mirror images, so the maxima are equal |
| Direction-aware GCN | 5.50 | 7.00 | Yes |

This is the mirror-symmetry limitation of Section 6.2 in numerical form:
the GCN produces the same graph embedding for a job and its mirror image,
so PPO cannot select different actions for them, whereas the
direction-aware layer produces different embeddings.

---

## 5. Experimental design

```
   +--------------------+        identical simulator, reward, action space,
   | CNN–PPO (baseline) |---+    PPO hyperparameters and training budget
   +--------------------+   |
                            +--> training --> evaluation on the same 500
   +--------------------+   |                 held-out routing jobs
   | GNN–PPO (proposed) |---+                          |
   +--------------------+                              v
                             success rate, routing cycles, invalid-action rate,
                             steps to convergence, training time, inference time
```

- **Controlled configuration:** each GNN configuration file inherits every
  setting from the CNN configuration and overrides only the encoder. This
  is enforced by `tests/test_policy_shapes.py`
  (`test_gnn_config_changes_only_the_encoder`).
- **Held-out evaluation:** 500 routing jobs generated from a separate
  random seed (20000) and evaluated with a deterministic policy. Jobs that
  time out are counted at the cycle limit k_max.
- **Ablations:** each ablation changes exactly **one** factor, so that any
  change in performance can be attributed to that factor.

| Configuration | Factor changed | Question addressed |
|---|---|---|
| CNN–PPO | none (baseline) | Reference performance of the published method |
| GCN + max pooling | encoder | Does the proposed method learn to route? |
| Direction-aware GCN + max pooling | message-passing layer | Is the GCN layer the limiting factor? |
| GCN + role-aware readout | readout | Is max pooling the limiting factor? |

---

## 6. Preliminary results

**Setup:** healthy 16×16 chips, one seed, 327,680 environment steps per
configuration, 500 identical held-out jobs.

| Configuration | Success | Mean cycles | Invalid actions per decision | Steps to convergence | Parameters |
|---|---|---|---|---|---|
| CNN–PPO (baseline) | 96.2% | 5.94 | 0.12 | 221 k | 2.15 M |
| GCN + max pooling | 4.0% | 25.06 | 0.88 | not reached | 8.6 k |
| **Direction-aware GCN + max pooling** | **99.4%** | **4.89** | **0.004** | **74 k** | 76 k |
| GCN + role-aware readout | 5.0% | 24.83 | 0.89 | not reached | 10 k |

Convergence is defined as a success rate of at least 95% in three
consecutive evaluations.

### 6.1 Summary

- The GCN with global max pooling, as initially specified, did not learn to
  route: 88% of its decisions were invalid actions.
- Replacing only the message-passing layer with the direction-aware GCN
  raised the success rate to 99.4%, above the CNN baseline.
- Replacing only the readout with a role-aware readout did not help (5.0%).
  The limiting factor is therefore the isotropic GCN layer, not max pooling.

### 6.2 Limitation of the isotropic GCN: mirror symmetry

The GCN aggregates neighbour messages with **the same weights**
irrespective of direction. Consider a routing job and its mirror image:

```
   Job A: goal to the east          Job B: mirror image of A

   . . . . . . . .                  . . . . . . . .
   . D D . . . G G                  G G . . . D D .
   . D D . . . G G                  G G . . . D D .
   . . . . . . . .                  . . . . . . . .

   correct action: E                correct action: W
```

1. **Identical node embeddings.** Each node in job B has the same
   neighbourhood as its mirror node in job A, reflected left to right.
   Because the GCN does not distinguish directions, mirrored nodes receive
   identical embeddings.
2. **Identical graph embedding.** Max pooling takes the maximum over all
   nodes, so both jobs produce the same graph embedding.
3. **Identical action distribution.** PPO receives the same input for both
   jobs and therefore outputs the same action probabilities, although the
   correct actions are opposite.

This property is verified by
`tests/test_graph_readout.py::test_isotropic_gcn_with_max_pooling_cannot_tell_a_job_from_its_mirror_image`,
and Section 4.7 (Steps 4 and 5) demonstrates it numerically.

### 6.3 Direction-aware message passing

The direction-aware GCN assigns **a separate weight matrix to each of the
eight directions**, so a message from the eastern neighbour is transformed
differently from one from the western neighbour:

```
   GCN                                  Direction-aware GCN

        W   W   W                            W_NW  W_N  W_NE
          \ | /                                  \  |  /
     W -- node -- W                      W_W -- node -- W_E
          / | \                                  /  |  \
        W   W   W                            W_SW  W_S  W_SE

   one weight matrix for all            one weight matrix per direction:
   directions: isotropic                direction-sensitive
```

- Only the message-passing layer changes; the graph and max pooling are
  identical.
- The success rate increased from **4.0% to 99.4%**, exceeding the CNN
  baseline (96.2%), with approximately 28 times fewer parameters and
  convergence in approximately one third of the environment steps (74 k
  versus 221 k).
- `tests/test_graph_readout.py::test_directional_layers_distinguish_the_mirror_image`
  verifies that mirrored jobs receive different embeddings.

### 6.4 Limitations

These results are from a single seed on a reduced chip size. They will be
repeated with five seeds on 30×30 chips, as in the base paper, before any
conclusion is drawn.

---

## 7. Implementation map

```
   observation --> graph_builder.py --> gnn.py -------------> graph_readout.py --> PPO
                   graph construction   graph convolution x3   global max pooling    actor + critic
                   (Sections 4.1-4.2)   (Sections 4.3-4.4)     (Section 4.5)         (Section 4.6)
```

| Component | Location |
|---|---|
| Simulator (chip, degradation, actions, reward) | `src/meda_routing/envs/` |
| CNN encoder (baseline) | `src/meda_routing/agents/cnn.py` |
| Graph construction (8-neighbour) | `src/meda_routing/representations/graph_builder.py` |
| GCN and direction-aware layers | `src/meda_routing/agents/gnn.py` |
| Max pooling and role-aware readout | `src/meda_routing/agents/graph_readout.py` |
| Controlled comparison on identical jobs | `src/meda_routing/experiments/method_comparison.py` |
| Experiment configurations (one per row of Section 5) | `configs/training/` |
| Experiment driver | `scripts/run_gnn_experiment.sh` |
| Remote execution on the GPU server | `scripts/run_on_gpu_server.sh` |
| Result tables and figures | `results/` |

The reduced experiment (16×16, one seed) runs in approximately one hour on
a four-core CPU:

```bash
SIZE=16 SEEDS=1 bash scripts/run_gnn_experiment.sh
```

The full experiment (30×30, five seeds) is intended for the GPU server:

```bash
GPU_SERVER=user@192.168.x.x bash scripts/run_on_gpu_server.sh bash scripts/run_gnn_experiment.sh
```
