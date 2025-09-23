from pathlib import Path

import numpy as np
import pandas as pd

from perceptron.perceptron_class import Perceptron
from viz.plot import (
    plot_data_scatter_2d,
    plot_decision_regions_2d,
    plot_learning_curve,
)


def main():
    # Build an absolute path to data/iris.data relative to this script
    DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "iris.data"
    df = pd.read_csv(DATA_PATH)

    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", 0, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # 1. raw data scatter
    plot_data_scatter_2d(
        X,
        y,
        labels=("Setosa", "Versicolor"),
        feature_names=("Sepal length [cm]", "Petal length [cm]"),
    )

    # 2. train perceptron
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    # 3. learning curve
    plot_learning_curve(ppn.errors_)

    # 4. decision regions
    plot_decision_regions_2d(
        X,
        y,
        classifier=ppn,
        feature_names=("Sepal length [cm]", "Petal length [cm]"),
    )


if __name__ == "__main__":
    main()
