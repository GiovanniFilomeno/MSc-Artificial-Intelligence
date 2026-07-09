# Selected Work — Evidence Map

This page is a selective guide to the work that currently has the clearest retained evidence. It distinguishes recorded results from independently reproduced benchmarks, and it states team ownership where that information is available. The broader curriculum and remaining archive are indexed separately in the [Course Catalog](course-catalog.md).

## Evidence Summary

| Project | Retained evidence | Recorded outcome | Ownership and limits |
| --- | --- | --- | --- |
| [MiniGrid DQN](<../Second Semester/Deep Reinforcement/Assignment 2>) | Executed training notebook, agent implementation, learning-curve figure, and ONNX-oriented evaluation script. | 50-episode rolling average score of **0.94 from episode 700 onward** in the stored training trace. | Coursework implementation. The metric is a notebook result, not an independent rerun; no compatible model artifact is published. |
| [CIFAR-10 Model Explanations](<../Third Semester/xAI/2nd Project>) | Executed explanation notebooks, utility code, visual outputs, environment file, and written report. | **82.59% test accuracy**; explanation findings cover class confusion, image regions, learned concepts, and temporal prediction flow. | Four-person team project; Giovanni Filomeno's reported contribution is **25%**. Detailed task-level attribution is not retained. |
| [Air-Quality Prediction](<../Fourth Semester/Programming in Python II/Assignment 6>) | Data pipeline, EDA code, model, training code, four-configuration result table, and CSV-only Shiny explorer. | Lowest validation MSE is **203.39**; that configuration's recorded test MSE is **626.78**. | Coursework implementation. Model selection is framed around validation, not the post-hoc minimum test score; no production or state-of-the-art claim is made. |
| [Projection-Space Exploration](<../Third Semester/xAI/1st Project>) | Analysis notebook, utilities, environment definition, and project overview. | Four projection methods compared over **1,000 sampled episodes per policy**; the analysis reports UMAP as the most informative for this task. | Research highlight with partial public evidence: some generated data and referenced figures are absent from the sanitized copy. |

## 1. MiniGrid DQN

### Engineering question

Can a value-based agent learn the partially observable MiniGrid DoorKey task from image observations within a 1,000-episode training run?

### What is implemented

- Replay-buffer sampling for off-policy learning.
- Epsilon-greedy exploration with scheduled decay.
- Separate online and target Q-networks.
- Double-DQN-style action selection for target computation.
- Soft target-network updates.
- A separate evaluation script for running multiple episodes.

### Evidence to inspect

- [Training notebook](<../Second Semester/Deep Reinforcement/Assignment 2/Minigrid_DQN_exercise_2024.ipynb>) — implementation, training trace, and learning curve.
- [Evaluation script](<../Second Semester/Deep Reinforcement/Assignment 2/minigrid_eval.py>) — original ONNX-oriented environment and 50-episode evaluation loop; no compatible ONNX artifact is retained.
- [Course overview](<../Second Semester/Deep Reinforcement/README.md>) — context within the broader deep-RL sequence.

The recorded rolling score rises from 0.00 at episode 0 to 0.94 by episode 700 and remains at 0.94 in each logged 50-episode checkpoint through episode 950. This is the strongest retained quantitative learning trace in the repository.

The standalone evaluator expects an ONNX file that is not published in this portfolio. It is retained as implementation evidence, not presented as a directly runnable evaluation package.

## 2. CIFAR-10 Model Explanations

### Research question

What image regions and learned concepts drive a CNN's predictions, and how do its class-level and instance-level errors evolve during training?

### What is implemented

- Gradient-based saliency maps.
- SHAP image explanations.
- Invertible concept-based explanations using non-negative matrix factorization.
- InstanceFlow analysis of prediction changes across epochs.

### Evidence to inspect

- [Executed combined notebook](<../Third Semester/xAI/2nd Project/submission-notebook.ipynb>) — retained visual outputs and analysis.
- [SHAP notebook](<../Third Semester/xAI/2nd Project/ex2_cifar10_shap/shap_analysis.ipynb>) — pixel-level contribution analysis.
- [Concept extractor](<../Third Semester/xAI/2nd Project/ConceptExtractor.py>) — concept-analysis implementation.
- [Written report](<../Third Semester/xAI/2nd Project/README.md>) — methods, results, limitations, and research conclusions.

The report records 82.59% test accuracy. Its most useful findings are diagnostic rather than purely numerical: cat images are frequently confused with dogs or frogs, higher-performing classes exhibit more specific concepts, and lower-performing classes rely more heavily on broad color patterns.

This is team work. The public report states a four-person team and a 25% contribution for Giovanni Filomeno; it does not identify which files or explanation methods were individually owned, so this portfolio does not claim sole authorship.

## 3. PM2.5 Prediction and Data Explorer

### Engineering question

How can a raw multiyear air-quality dataset be turned into a time-aware regression workflow and an interactive inspection tool?

### What is implemented

- Dataset download, archive handling, cleaning, interpolation, and datetime indexing.
- Exploratory trend, correlation, and distribution visualizations.
- Chronological 80/10/10 train, validation, and test splitting.
- Training-only feature scaling and PyTorch `DataLoader` construction.
- Configurable MLP regression and four-configuration experiment comparison.
- CSV-only Shiny explorer for pollutant visualization and rolling-mean inspection.

### Evidence to inspect

- [Project README](<../Fourth Semester/Programming in Python II/Assignment 6/README.md>) — component-level map.
- [Preprocessing](<../Fourth Semester/Programming in Python II/Assignment 6/a6_ex1.py>) and [data loaders](<../Fourth Semester/Programming in Python II/Assignment 6/a6_ex3.py>) — data-engineering workflow.
- [Training](<../Fourth Semester/Programming in Python II/Assignment 6/a6_ex5.py>) and [experiment comparison](<../Fourth Semester/Programming in Python II/Assignment 6/a6_ex6.py>) — modelling workflow.
- [Recorded results](<../Fourth Semester/Programming in Python II/Assignment 6/a6_ex6.txt>) — four experiment configurations.
- [Shiny application](<../Fourth Semester/Programming in Python II/Assignment 6/app.py>) — interactive interface.

The lowest validation MSE is 203.39 for the `(256, 128)` configuration; its recorded test MSE is 626.78. The archival table also contains test MSE values of 646.30, 589.64, 626.78, and 633.74. Although 589.64 is the smallest of those test values, using that post-hoc minimum for model selection would leak test information. Because the archive also contains no explicit baseline, these values demonstrate experiment execution rather than superiority over a simpler model.

The public explorer no longer accepts serialized model or pickle uploads; it requires a parseable datetime column and numeric series, and keeps training separate from interactive inspection. The [project review](<../Fourth Semester/Programming in Python II/Assignment 6/README.md>) also documents preprocessing leakage and test-set-selection limitations found during this portfolio audit.

## Research Highlight: Projection-Space Exploration

This project examines Cliff Walking trajectories produced by Sarsa, Q-learning, Expected Sarsa, and a random policy. It compares PCA, ICA, t-SNE, and UMAP and adds algorithm, trajectory-stage, state, action, and reward metadata to the visual analysis.

The [analysis notebook](<../Third Semester/xAI/1st Project/A1_submission_notebook.ipynb>) describes a sample of 1,000 episodes per policy. It concludes that nonlinear methods are more useful for these trajectories and that UMAP provides the clearest balance of local and global structure.

The public edition does not include all generated arrays and rendered figures referenced by the notebook. The project is included because the research design and interpretation are valuable, but its evidence status is intentionally described as partial.

## Deliberately Not Featured

- **SAT solver suite:** retained in the [coursework archive](<../Fourth Semester/SAT/Project>), but on hold until its solvers agree with the reference implementation across the benchmark set. It should not be used as performance evidence yet.
- **Computability and Complexity:** the public folder currently contains an index description, not the underlying NP-completeness report.
- **Robopsychology:** the public folder currently contains an index description, not the written reviews it summarizes.
- **Broad course sequences:** Deep Learning, Computer Vision, Probabilistic Models, recommender systems, and other subjects provide valuable curriculum context, but only the retained artifacts described in the catalog should be treated as repository evidence.

## Continue Through the Archive

- [Complete Course Catalog](course-catalog.md)
- [First Semester](<../First Semester>)
- [Second Semester](<../Second Semester>)
- [Third Semester](<../Third Semester>)
- [Fourth Semester](<../Fourth Semester>)
- [Additional Courses](<../Additional Courses>)
