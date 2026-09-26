# Data

This directory contains the dependency-graph dataset used in the project.

## Dataset

- **Target package:** Gatsby 5.16.1
- **Ecosystem:** npm
- **Source:** [deps.dev](https://deps.dev/) (Google Open Source Insights)
- **File:** `gatsby_5.16.1_dependencies_labeled.json`

The dependency graph was derived from deps.dev resolved dependency data for Gatsby 5.16.1.

## Attribution and License

deps.dev-generated data is made available under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

- deps.dev: https://deps.dev/
- deps.dev source repository: https://github.com/google/deps.dev
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

This repository includes a **modified version** of the dependency data. The original dependency information was supplemented for this project with synthetic experiment fields, including `infection_label` and related metadata.

## Synthetic Infection Labels

The `infection_label` values in this dataset are **not real-world malware or compromise records**.

They were created as a controlled synthetic benchmark for evaluating the candidate-ranking and blocking-optimization pipeline. In particular:

- the labels are used only for experimental evaluation;
- they do not represent observed security incidents;
- they should not be interpreted as real compromise ground truth;
- results based on these labels do not establish real-world detection accuracy or quantum advantage.

The known initial infected source used in the experiments is:

```text
es-errors@1.3.0
```

## Graph Interpretation

The raw dependency relation follows the package dependency direction. For propagation analysis, the project reverses dependency edges so that potential compromise can flow from a dependency toward packages that depend on it.

## Reproducibility Note

The dataset in this repository is intended to reproduce the experiments reported in this project. If you replace it with newly retrieved deps.dev data, graph structure and downstream results may differ because package metadata can evolve over time.
