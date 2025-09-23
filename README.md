# perceptron-playground

[![CI](https://github.com/grthomson/perceptron-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/grthomson/perceptron-playground/actions/workflows/ci.yml)

Experiments with perceptrons with a view to entity resolution modelled as binary classification. Currently running a simple single-layer model for i) classification of the classic Iris dataset and ii) the entity resolution task of matching a toy dataset with an error-imputed version of itself.

## Acknowledgements

This project is heavily based on examples from the excellent book
[**Python Machine Learning, 3rd Edition**, by Sebastian Raschka](https://github.com/rasbt/python-machine-learning-book-3rd-edition).

## Entity Resolution (Record Linkage) — Quick Start

```bash
# 0) Setup (create venv, install deps, enable pre-commit)
python -m venv .venv
# Git Bash:   source .venv/Scripts/activate
# PowerShell: .venv\Scripts\Activate.ps1
# cmd.exe:    .venv\Scripts\activate.bat
pip install -e ".[dev]"
pre-commit install

# 1) (Optional) Sanity-check with the classic Iris demo
python scripts/train_and_run_iris_perceptron.py

# 2) Generate a small synthetic people dataset (deterministic with Faker)
python scripts/make_toy_clean_dataset.py            # -> data/toy_people_clean.csv

# 3) Create a noisy “duplicate” copy + alignment labels
python scripts/make_noisy_copy.py                   # -> data/toy_people_noisy.csv, data/toy_labels.csv

# 4) Train a perceptron for entity resolution (Levenshtein similarity features)
python scripts/train_linkage_perceptron.py --cols forename,surname,address,city,postcode
# -> saves model to data/models/linkage_perceptron.pkl and prints weights/updates

# 5) Score pairs with the trained model (writes a scored pairs CSV)
python scripts/run_linkage_perceptron.py            # -> data/toy_scored_pairs.csv
```

## What those steps do

- Setup: creates an isolated Python environment, installs project + dev tooling, enables pre-commit checks locally.

- Iris demo (optional): trains on two Iris features and pops three plots (scatter, convergence, decision regions) to validate basic single layer perceptron functionality.

- Make toy data: generates the dataset toy_people_clean.csv with names/addresses using the Faker package (by default seeded for reproducibility).

- Make noisy copy: introduces realistic mistakes (typos, case changes, postcode spacing, etc.) to create toy_people_noisy.csv.

- Writes toy_labels.csv, which associates each clean row with its corresponding noisy row (these are actual matches or true positives - all other row pairs are actual non-matches or true negatives).

- Train entity resolution perceptron: builds normalized Levenshtein similarity features for the selected columns, then learns a linear decision rule (weights + bias).

- Score pairs: applies the trained entity resolution model to candidate pairs (by default, the full cartesian product of the toy dataset and its noisy copy). Writes toy_scored_pairs.csv with a score and prediction per pair.

- Note: generated CSVs and model artifacts under data/ are ignored by git (see .gitignore). The code to recreate them lives in scripts/.

## Visualisations

Plotting helpers are kept in ```src/viz/plot.py```.

Classic iris dataset outputs (code heavily borrowed from [Raschka](https://github.com/rasbt/python-machine-learning-book-3rd-edition))

### Raw Data Scatter
![Iris scatter](docs/images/iris_scatter.png)

### Learning Curve
```
from viz.plot import plot_learning_curve
plot_learning_curve(ppn.errors_)
```
Shows convergence: number of weight updates per epoch.
![Iris learning curve](docs/images/iris_learning.png)

### 2D Decision Regions
```
from viz.plot import plot_decision_regions_2d
plot_decision_regions_2d(
    X, y, classifier=ppn, feat_idx=(0, 1),
    feature_names=("forename_sim", "surname_sim"),
)
```
Linear decision boundary learned by the perceptron on two Iris features.
![Iris decision regions](docs/images/iris_decision_regions.png)

## Entity Resolution Visualisations

### 3D decision plane (only when training on exactly three features):

```
from viz.plot import plot_decision_plane_3d
plot_decision_plane_3d(
    X, y, classifier=ppn, feat_idx=(0, 1, 4),
    feature_names=("forename_sim", "surname_sim", "postcode_sim"),
)
```

![Entity resolution decision regions](docs/images/entity_resolution_3dim_boundary.png)

### 2D decision regions (choose any two features, e.g. forename_sim vs surname_sim):

For models with 4–5 features, the decision boundary cannot be visualised in 3 dimensional Euclidean space. Current plan is to add a Principal Component Analysis (PCA) projection step and provide some meaningful plot using the reduced dimensions.
