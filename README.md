# Giovanni Filomeno — AI/ML Engineering Portfolio

MSc Artificial Intelligence coursework at Johannes Kepler University Linz (JKU).

I build and evaluate machine-learning systems across deep reinforcement learning, model interpretability, and applied ML. This repository is the public, curated edition of my MSc coursework: featured projects are presented first, while a semester-by-semester curriculum map and the strongest remaining artifacts provide academic context.

## Featured Projects

### MiniGrid DQN — Deep Reinforcement Learning

Implemented a value-based agent for the partially observable MiniGrid DoorKey environment, including replay memory, epsilon-greedy exploration, online and target networks, and soft target updates. The executed training notebook records a 50-episode rolling average score of **0.94 from episode 700 onward**.

- **Evidence:** [project files](<Second Semester/Deep Reinforcement/Assignment 2>) · [executed training notebook](<Second Semester/Deep Reinforcement/Assignment 2/Minigrid_DQN_exercise_2024.ipynb>) · [evaluation script](<Second Semester/Deep Reinforcement/Assignment 2/minigrid_eval.py>)
- **Stack:** Python, PyTorch, Gymnasium, MiniGrid, NumPy, ONNX evaluation
- **Scope:** Coursework implementation. The reported score is the rolling training result stored in the notebook, not a newly reproduced benchmark. The ONNX-oriented evaluator is retained as source, but no compatible model artifact is published.

### Explaining a CIFAR-10 Classifier — Explainable AI

Analyzed a custom CNN with four complementary explanation approaches: saliency maps, SHAP, invertible concept-based explanations, and InstanceFlow. The project reports **82.59% CIFAR-10 test accuracy** and uses the explanations to investigate class confusion, learned visual concepts, and training-time prediction behavior.

- **Evidence:** [project files](<Third Semester/xAI/2nd Project>) · [executed analysis notebook](<Third Semester/xAI/2nd Project/submission-notebook.ipynb>) · [project report](<Third Semester/xAI/2nd Project/README.md>)
- **Stack:** Python, PyTorch, CIFAR-10, SHAP, non-negative matrix factorization, visual analytics
- **Scope:** Four-person JKU team project. The public report records Giovanni Filomeno's contribution as **25%**; it does not preserve a more detailed task-level attribution.

### PM2.5 Prediction and Air-Quality Explorer — Applied ML

Built an end-to-end workflow around the Beijing Multi-Site Air Quality dataset: download and cleaning, time-aware splitting, exploratory analysis, feature scaling, PyTorch regression, hyperparameter experiments, and an interactive Shiny data explorer. Across the four recorded configurations, the lowest validation MSE is **203.39**; that validation-selected configuration has a recorded test MSE of **626.78**.

- **Evidence:** [project overview](<Fourth Semester/Programming in Python II/Assignment 6/README.md>) · [source and artifacts](<Fourth Semester/Programming in Python II/Assignment 6>) · [recorded experiment table](<Fourth Semester/Programming in Python II/Assignment 6/a6_ex6.txt>)
- **Stack:** Python, pandas, scikit-learn, PyTorch, Matplotlib, Shiny
- **Scope:** Coursework implementation. A different configuration has the lowest post-hoc test MSE, but it is not promoted as the selected model because choosing on test results would leak test information. No production or state-of-the-art claim is made.

## Research Highlight

### Projection-Space Exploration of RL Trajectories

Compared PCA, ICA, t-SNE, and UMAP for visualizing trajectories from Sarsa, Q-learning, Expected Sarsa, and a random policy in Cliff Walking. The analysis samples **1,000 episodes per policy** and reports that UMAP provided the most useful balance of local and global structure for this visualization task.

- **Evidence:** [project overview](<Third Semester/xAI/1st Project/README.md>) · [analysis notebook](<Third Semester/xAI/1st Project/A1_submission_notebook.ipynb>)
- **Artifact note:** The public copy retains the analysis and utility code, but not every generated dataset or rendered figure referenced by the notebook. It is therefore presented as a research highlight rather than as one of the three primary reproducible examples.

For a concise evidence and ownership map, see [Selected Work](docs/showcase.md).

## Retained Technical Evidence

| Capability | Evidence in this repository |
| --- | --- |
| Deep learning and reinforcement learning | Executed DQN training, a PPO coursework notebook, NumPy neural-network components, CNN explanation workflows, and PyTorch regression. |
| Explainability and visual analytics | Saliency, SHAP, concept-based explanations, dimensionality reduction, trajectory analysis, and model-behavior visualization. |
| Applied ML engineering | Data ingestion and cleaning, temporal splitting, feature scaling, experiment comparison, evaluation, and interactive application development. |
| Statistical and symbolic foundations | Bayesian-network utilities and notebooks, stochastic-simulation code, recommender modules, Prolog/ILP work, and SAT/SMT source—with the custom SAT solvers explicitly held from performance claims. |

## Coursework Archive

The retained material follows the original academic structure so that artifacts can be inspected in course context.

| Period | Coverage |
| --- | --- |
| [First Semester](<First Semester>) | Supervised learning, neural-network foundations, sequence models, planning and reasoning, and reinforcement learning. |
| [Second Semester](<Second Semester>) | Advanced deep learning, deep reinforcement learning, unsupervised and theoretical ML, stochastic simulation, and knowledge representation. |
| [Third Semester](<Third Semester>) | Probabilistic models, computer vision, explainable AI, computability and complexity, and AI communication. |
| [Fourth Semester](<Fourth Semester>) | Statistics, recommender systems, Python engineering, SAT programming, life-science AI, and stochastic simulation. |
| [Additional Courses](<Additional Courses>) | Biological sequence analysis and computational physics. |

Use the [Course Catalog](docs/course-catalog.md) for the complete assignment-level map. The catalog documents curriculum breadth; the featured projects above are the recommended starting point for evaluating retained implementation and analytical work.

## Public-Edition and Validation Notes

This is a sanitized public edition of an academic working repository. Student identifiers and private submission metadata were removed, and some original datasets, reports, generated outputs, checkpoints, or course-specific dependencies are not included. Individual folders may therefore be inspectable without being fully reproducible as standalone packages.

The C++ SAT solver coursework remains in the archive, but it is **deliberately excluded from featured work** while cross-solver correctness discrepancies are being resolved and validated against a reference solver. Portfolio claims should be based on verified behavior, so no performance claim is made for that project here.

## Rights, Attribution, and Reuse

Course prompts, templates, third-party datasets, papers, and external libraries remain under their original ownership and licenses. Authored code and analysis are shared for portfolio review and learning; please do not submit this material as your own coursework. See [NOTICE.md](NOTICE.md) for additional attribution and reuse notes.
