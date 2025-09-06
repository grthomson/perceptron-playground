import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Import from the package under src/
from perceptron.perceptron_class import Perceptron, plot_decision_regions


def main():
    # Build an absolute path to data/iris.data relative to this script
    DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "iris.data"
    df = pd.read_csv(DATA_PATH)
    df.tail()

    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", 0, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # plot data
    plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="Setosa")
    plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="s", label="Versicolor")

    plt.xlabel("Sepal length [cm]")
    plt.ylabel("Petal length [cm]")
    plt.legend(loc="upper left")
    plt.show()

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Number of updates")
    plt.show()

    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel("Sepal length [cm]")
    plt.ylabel("Petal length [cm]")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
