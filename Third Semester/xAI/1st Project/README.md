# Projection Space Exploration

This project studies reinforcement-learning trajectories from the Cliff Walking environment by projecting high-dimensional sequential states into two-dimensional visual spaces. The goal is to compare how different temporal-difference policies move through the state space and whether their behaviour becomes separable after dimensionality reduction.

## What Was Built

- Analyzed trajectory data for Sarsa, Q-learning, Expected Sarsa, and a random policy.
- Compared policy behaviour using projected trajectory paths.
- Explored PCA, ICA, t-SNE, and UMAP as alternative projection techniques.
- Added metadata to the visualizations so projected paths could be interpreted by algorithm, state, action, reward, and trajectory segment.
- Included additional notebooks for 2048 and Rubik-style projection examples.

## Skills Demonstrated

- Reinforcement-learning trajectory analysis.
- Dimensionality reduction for sequential/state-space data.
- Visual analytics for model and policy behaviour.
- Jupyter-based exploratory research workflows.
- Shared utility code and a documented Conda/Docker environment.

## Repository Notes

The public portfolio copy keeps the notebooks, utility code, and environment files. The Cliff Walking trajectory arrays and cached projections are not included, so the main analysis notebook is inspectable but is not self-contained for a clean re-run.

## Main Files

| File | Purpose |
| --- | --- |
| `A1_submission_notebook.ipynb` | Main analysis notebook for the Cliff Walking projection study. |
| `solution_2048.ipynb` | Additional projection-space example for 2048 trajectories. |
| `solution_rubik.ipynb` | Additional projection-space example for cube-state trajectories. |
| `utils.py` | Shared helpers for loading trajectories, caching projections, and plotting. |
