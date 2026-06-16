# MSc Artificial Intelligence Coursework Portfolio

This repository is a portfolio archive of my MSc in Artificial Intelligence at Johannes Kepler University Linz (JKU). It collects assignments, implementation projects, reports, notebooks, and experiments across machine learning, deep learning, reinforcement learning, computer vision, explainable AI, probabilistic modelling, SAT solving, recommender systems, AI for life sciences, and stochastic simulation.

The original semester-by-semester structure is preserved so the academic context stays clear. This README and the `docs/` folder provide the curated GitHub layer for visitors who want to understand the work quickly.

## Portfolio Highlights

| Area | Representative work | What it demonstrates |
| --- | --- | --- |
| SAT solving and symbolic AI | [SAT Programming Project](<Fourth Semester/SAT/Project>) | C++17 implementations of recursive solving, watched literals, CDCL-style solving, DIMACS-style parsing, and PicoSAT integration. |
| Explainable AI | [Projection Space Exploration](<Third Semester/xAI/1st Project>) and [Model Explanations](<Third Semester/xAI/2nd Project>) | Trajectory analysis, dimensionality reduction, saliency maps, SHAP, and concept-based explanations for neural models. |
| Deep reinforcement learning | [Imitation Learning](<Second Semester/Deep Reinforcement/Assgnment 1>), [Minigrid DQN](<Second Semester/Deep Reinforcement/Assignment 2>), and [PPO](<Second Semester/Deep Reinforcement/Assignment 3>) | Policy learning, value-based control, imitation learning, evaluation scripts, and trained model artifacts. |
| Computer vision | [Computer Vision Assignments](<Third Semester/Computer Vision>) | Image processing, segmentation, optical flow, object tracking, detection, and 3D reconstruction. |
| Deep learning | [Deep Learning II](<Second Semester/Deep Learning II>) and [LSTM](<First Semester/LSTM>) | PyTorch workflows for CNNs, VAEs, adversarial training, normalizing flows, Bayesian deep learning, recurrent networks, and transformers. |
| Recommender systems | [Learning from User-generated Data](<Fourth Semester/Learning from User-generated Data>) | Popularity baselines, item-kNN, matrix factorization, evaluation metrics, calibration, and content-based filtering. |
| AI in life sciences | [Artificial Intelligence in Life Science](<Fourth Semester/Artificial Intelligence in Life Science>) | QSAR modelling, molecular generation, synthesis prediction, and scientific ML reporting. |
| Statistical and probabilistic AI | [Probabilistic Models](<Third Semester/Probabilistic Models>), [Statistics for AI](<Fourth Semester/Statistics for AI>), and [Stochastic Simulation](<Fourth Semester/Stochastic Simulation>) | Probabilistic inference, sampling, statistical modelling, simulation, Monte Carlo methods, and stochastic processes. |

For a more curated tour, see [Selected Work](docs/showcase.md). For the full academic map, see [Course Catalog](docs/course-catalog.md).

## Repository Map

| Folder | Contents |
| --- | --- |
| [First Semester](<First Semester>) | Foundations in supervised learning, neural networks, LSTMs, planning and reasoning, and reinforcement learning. |
| [Second Semester](<Second Semester>) | Advanced deep learning, deep reinforcement learning, unsupervised learning, theoretical ML, stochastic simulation, knowledge representation, and human-centered AI coursework. |
| [Third Semester](<Third Semester>) | Probabilistic models, computer vision, explainable AI, computability and complexity, and AI communication. |
| [Fourth Semester](<Fourth Semester>) | Statistics for AI, recommender systems, Python engineering, SAT programming, AI for life sciences, and stochastic simulation. |
| [Additional Courses](<Additional Courses>) | Biological sequence analysis and computational physics. |
| [docs](docs) | Public-facing portfolio notes, course catalog, and repository hygiene guidance. |

## Technical Stack

The coursework spans Python, Jupyter, PyTorch, NumPy, pandas, scikit-learn, Matplotlib, C++17, SAT/SMT tooling, probabilistic modelling utilities, recommender-system pipelines, and scientific reporting workflows.

## How To Browse

Start with [Selected Work](docs/showcase.md) if you are evaluating the portfolio. Use [Course Catalog](docs/course-catalog.md) if you want a semester-by-semester view of the MSc curriculum. Individual assignment folders usually contain the submitted notebook, report, source code, data artifacts, or model artifacts needed for that specific exercise.

Some notebooks depend on the original course environments and datasets. Where possible, code, reports, and outputs are kept together to make the reasoning and results inspectable even when the exact environment is not reproduced.

## Repository Hygiene

This repository is an academic archive rather than a single production package. It contains notebooks, reports, datasets, trained models, exported submissions, and generated outputs. The `.gitignore` and `.gitattributes` files are configured to keep future generated artifacts cleaner and to make GitHub language statistics less noisy.

See [Repository Hygiene](docs/repository-hygiene.md) for the current size notes, large-artifact guidance, and suggested steps before publishing a lean public mirror.

## Academic Integrity And Reuse

The work is shared as a portfolio and learning archive. Course prompts, third-party datasets, papers, libraries, and assignment templates remain under their original ownership and licenses. Please do not submit this material as your own coursework.

See [Notice](NOTICE.md) for reuse and third-party material notes.
