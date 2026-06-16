# Projection Space Exploration

This project studies reinforcement-learning trajectories from the Cliff Walking environment by projecting high-dimensional sequential states into two-dimensional visual spaces. The goal is to compare how different temporal-difference policies move through the state space and whether their behaviour becomes separable after dimensionality reduction.

## What Was Built

- Generated trajectory data for Sarsa, Q-learning, Expected Sarsa, and a random policy.
- Compared policy behaviour using projected trajectory paths.
- Explored PCA, ICA, t-SNE, and UMAP as alternative projection techniques.
- Added metadata to the visualizations so projected paths could be interpreted by algorithm, state, action, reward, and trajectory segment.
- Included additional notebooks for 2048 and Rubik-style projection examples.

## Skills Demonstrated

- Reinforcement-learning trajectory analysis.
- Dimensionality reduction for sequential/state-space data.
- Visual analytics for model and policy behaviour.
- Jupyter-based exploratory research workflows.
- Clear separation between source notebooks and generated datasets/plots.

## Repository Notes

The public portfolio copy keeps notebooks and utility code, but omits generated trajectory arrays, cached projections, exported HTML, and presentation artifacts. Re-running the notebooks may regenerate local `data/`, `cache/`, and `export/` folders; those paths are intentionally ignored for GitHub hygiene.

## Main Files

| File | Purpose |
| --- | --- |
| `A1_submission_notebook.ipynb` | Main analysis notebook for the Cliff Walking projection study. |
| `solution_2048.ipynb` | Additional projection-space example for 2048 trajectories. |
| `solution_rubik.ipynb` | Additional projection-space example for cube-state trajectories. |
| `utils.py` | Shared helpers for loading trajectories, caching projections, and plotting. |
