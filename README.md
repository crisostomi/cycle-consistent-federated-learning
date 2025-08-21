# cycle-consistent-federated-learning

Code to reproduce the experiment "B.5 Federated Learning" in [Cycle-Consistent Model Merging]((https://arxiv.org/abs/2405.17897)) (NeurIPS 2024).

[!WARNING] 21th August 2025: code is not working. Missing a part that was directly modified in `flwr` due to rebuttal haste. Will try to fix it.

## Install dependencies

```bash
    uv sync
```

## Run (Simulation Engine)

In the `cycle-consistent-federated-learning` directory, use `flwr run` to run a local simulation:

```bash
    flwr run
```
