# Assignment 6 - Air-quality Prediction

This assignment is a small applied machine-learning project for PM2.5 prediction using the Beijing Multi-Site Air Quality dataset.

| Component | What it does | Skills demonstrated |
| --- | --- | --- |
| `a6_ex1.py` | Downloads/extracts the dataset, selects a station, cleans missing/out-of-range values, builds a datetime index, and exports `air_quality_cleaned.csv`. | Data ingestion, zip handling, pandas preprocessing, time-indexed data cleaning. |
| `a6_ex2.py` | Generates exploratory plots for PM2.5 trends, correlations, and histograms. | EDA, Matplotlib plotting, statistical visualization. |
| `a6_ex3.py` | Splits the cleaned dataset into train/validation/test loaders and saves a scaler. | Feature scaling, temporal splits, PyTorch DataLoader construction. |
| `a6_ex4.py` | Defines an MLP regressor for PM2.5 prediction. | PyTorch module design, configurable hidden layers, dropout. |
| `a6_ex5.py` | Trains the model, reports losses, saves weights, and creates prediction plots. | Training loops, validation, test-set prediction, model persistence. |
| `a6_ex6.py` | Runs a small hyperparameter search over network size, dropout, learning rate, and epochs. | Experiment management, model comparison, result logging. |
| `app.py` | Application entry point for using the trained model workflow. | Packaging project logic into a runnable app script. |

Portfolio takeaway: this assignment is a good applied example because it connects data engineering, exploratory analysis, neural-network regression, model evaluation, and artifact generation in one workflow.
