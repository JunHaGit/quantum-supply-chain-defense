# Quantum Supply Chain Defense

A two-stage prototype for software supply-chain defense using **Continuous-Time Quantum Walk (CTQW)** for candidate risk ranking and **QUBO/QAOA** for blocking-node optimization.

This repository reconstructs the technical pipeline developed for the **Quantum Reframing Challenge 2026**. The project reframes software supply-chain incident response as:

1. **Candidate detection:** reduce a dependency graph to a ranked set of potentially high-risk packages.
2. **Blocking optimization:** select a fixed number of packages to block while balancing infection reduction and service availability.

> **Scope note:** this is a research / competition prototype. The dependency graph is based on real package dependency data, but the embedded infection labels are synthetic and are not real-world compromise records. Results should not be interpreted as evidence of real-world detection accuracy or quantum advantage.

---

## Architecture

```text
deps.dev dependency graph
        |
        v
candidate_detection.py
        |
        |  CTQW / BFS / PageRank / Betweenness / Monte Carlo
        v
outputs/rankings.csv
        |
        v
blocking_optimization.py
        |
        |  QUBO + QAOA / Greedy / Simulated Annealing / Betweenness
        v
stage2_outputs/
```

A separate `ctqw_hardware_demo.py` provides a reduced-scale CTQW circuit demo that can run either on Qiskit Aer or, optionally, IBM Quantum hardware.

---

## Repository Structure

```text
quantum-supply-chain-defense/
├── README.md
├── requirements.txt
├── candidate_detection.py
├── blocking_optimization.py
├── ctqw_hardware_demo.py
├── data/
│   └── gatsby_5.16.1_dependencies_labeled.json
└── docs/
    └── final_presentation.pdf
```

Generated experiment outputs are written to `outputs/` and `stage2_outputs/`.

---

## Data

The dependency graph is derived from **deps.dev** data for **Gatsby 5.16.1**.

The supplied JSON contains:

- **1,387 raw nodes**
- **2,851 raw dependency edges**
- package/version information for NPM dependencies
- a synthetic `infection_label` used only for evaluation

The Stage 1 loader collapses duplicate `package@version` entries before ranking.

### Important label limitation

The `infection_label` field is a **synthetic toy benchmark**, not observed malware ground truth. The labels were constructed for a controlled experimental setting and therefore:

- are **not** used as an input to the ranking algorithms;
- are read only after ranking is complete, for evaluation;
- do **not** establish real-world compromise-detection accuracy;
- do **not** establish quantum advantage.

The default known source package is:

```text
es-errors@1.3.0
```

Dependency edges are reversed when constructing the propagation graph because a compromised dependency may affect packages that depend on it.

---

## Stage 1 — Candidate Detection

`candidate_detection.py` compares several graph-based and quantum-inspired ranking methods:

- BFS-based ranking
- Personalized PageRank
- Betweenness Centrality
- Monte Carlo infection-risk simulation
- Directional CTQW risk heuristic

The CTQW ranking uses a Hermitian, phase-aware Hamiltonian derived from the directed propagation graph. Its evolution is evaluated with **Qiskit Statevector simulation on a classical computer**.

The embedded infection labels are excluded from graph construction and score computation. They are used only after ranking to calculate Top-K precision, recall, and F1.

### Run Stage 1

From the repository root:

```bash
python candidate_detection.py
```

Useful options:

```bash
python candidate_detection.py \
  --top-k 50 \
  --candidate-limit 0 \
  --ctqw-time-max 20 \
  --ctqw-time-steps 200 \
  --monte-carlo-simulations 2000
```

`--candidate-limit 0` uses all packages reachable from the known infection source.

### Stage 1 outputs

By default, outputs are written to:

```text
outputs/
├── rankings.csv
├── stage1_predictions_with_labels.csv
├── method_comparison.csv
├── infection_ground_truth.csv
└── experiment_summary.json
```

`outputs/rankings.csv` is the main handoff file to Stage 2. It intentionally excludes the ground-truth `infection_label`.

---

## Stage 2 — Blocking Optimization

`blocking_optimization.py` reads the Stage 1 ranking and selects a fixed number of packages to block.

The default pipeline is:

```text
Stage 1 ranked candidates: 32
        ↓
QUBO/QAOA candidate set: 15
        ↓
Packages to block: 5
```

All three counts are configurable from the command line.

### QUBO formulation

For each candidate package, the code estimates:

- infection reduction from blocking that package alone;
- pairwise interaction / overlap when blocking two packages together;
- an availability-cost proxy based on graph in-degree.

These terms form a pairwise QUBO approximation. Final solutions are evaluated again with a full deterministic BFS propagation simulation rather than relying only on QUBO energy.

### QAOA implementation

The QAOA implementation uses:

- fixed-Hamming-weight initialization;
- a ring **XY mixer** implemented with `RXX` and `RYY`;
- one or more QAOA layers;
- COBYLA parameter optimization;
- Qiskit Aer MPS simulation by default.

The fixed-Hamming-weight state and XY mixer preserve the requirement that exactly `K` packages are selected.

The current implementation uses a deterministic feasible initial state consisting of the first `K` candidates. It should **not** be described as a Greedy warm start.

### Classical comparisons

Stage 2 also evaluates:

- Greedy selection
- Simulated Annealing
- Betweenness Centrality

For sufficiently small search spaces, an exhaustive oracle is also computed as a reference.

### Run Stage 2

Run Stage 1 first so that `outputs/rankings.csv` exists:

```bash
python candidate_detection.py
python blocking_optimization.py
```

The default Stage 2 configuration corresponds to a **15-candidate / 5-block** experiment.

A **20-candidate / 10-block** experiment can be run with:

```bash
python blocking_optimization.py \
  --stage1-candidate-count 32 \
  --candidate-count 20 \
  --block-count 10 \
  --qaoa-reps 1 \
  --qaoa-maxiter 100 \
  --qaoa-restarts 5 \
  --shots 10000
```

### Stage 2 outputs

By default:

```text
stage2_outputs/
├── method_comparison.csv
├── selected_nodes.csv
├── qubo_candidates.csv
└── experiment_summary.json
```

---

## IBM Quantum Execution

### CTQW hardware demo

`ctqw_hardware_demo.py` is a reduced-scale hardware feasibility demo.

Its default execution uses **Qiskit Aer**, so no IBM Quantum account is required:

```bash
python ctqw_hardware_demo.py
```

To use IBM Quantum hardware:

```bash
python ctqw_hardware_demo.py --hardware
```

IBM Quantum credentials must already be configured locally through Qiskit. **No API token or IBM Quantum instance identifier should be stored in this repository.**

### QAOA hardware sampling

Stage 2 can optionally optimize QAOA parameters with Aer and submit the final bound circuit to IBM Quantum hardware:

```bash
python blocking_optimization.py \
  --qaoa-backend ibm \
  --confirm-qpu
```

A backend can also be specified explicitly with `--ibm-backend`.

This mode is **hybrid**:

```text
Aer parameter optimization
        ↓
final bound QAOA circuit
        ↓
IBM QPU sampling
```

It does not perform the full variational optimization loop on the QPU.

---

## Installation

Python **3.10+** is recommended.

Create and activate a virtual environment, then install:

```bash
pip install -r requirements.txt
```

The project depends on:

- NumPy
- SciPy
- NetworkX
- Matplotlib
- Qiskit
- Qiskit Aer
- Qiskit IBM Runtime

Package versions are intentionally left unpinned because the original experiment environment did not preserve a complete lock file.

---

## Experimental Interpretation

The project should be interpreted as an **end-to-end feasibility prototype**, not as a production security system.

The main technical contribution is the integration of:

```text
dependency-network modeling
        +
CTQW-based candidate ranking
        +
QUBO formulation
        +
QAOA / classical blocking optimization
        +
post-block propagation evaluation
```

The project presentation reports experiments across multiple candidate/blocking sizes, including 15-to-5 and 20-to-10 cases. Candidate counts and other parameters were varied during development, so intermediate CSV files from different runs may contain different numbers of nodes.

---

## Limitations

- Infection labels are synthetic rather than real incident labels.
- The benchmark is a controlled toy setting and is not representative of the full software-package ecosystem.
- Stage 1 CTQW ranking is simulated classically.
- Hardware experiments operate on reduced or final-bound circuits because of current QPU constraints.
- QUBO captures single-node and pairwise effects; higher-order blocking interactions are only reflected in the final BFS evaluation.
- Classical and quantum runtimes are not directly comparable, and this repository does not claim quantum speedup.
- Larger candidate sets quickly make exhaustive verification impractical.

---

## Project Context

This project was developed as a team submission for the **Quantum Reframing Challenge 2026** in the software supply-chain security track.

The project explored whether a software dependency-network response problem could be reframed into a two-stage quantum workflow:

1. CTQW-based high-risk candidate discovery
2. QUBO/QAOA-based blocking-set optimization

See `docs/final_presentation.pdf` for the original project framing, experiment design, and competition presentation.

---

## Security

Do not commit API keys, IBM Quantum tokens, account instance identifiers, or other credentials.

IBM Quantum authentication should be configured outside the repository using Qiskit's local credential management.
