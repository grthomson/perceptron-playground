import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron:
    """Perceptron classifier.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Passes over the training dataset.
    random_state : int
        RNG seed for weight initialization.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    b_ : float
        Bias unit after fitting.
    errors_ : list[int]
        Number of misclassifications (updates) in each epoch.
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_: np.ndarray | None = None
        self.b_: float | None = None
        self.errors_: list[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Fit training data."""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X: np.ndarray) -> np.ndarray | float:
        """Calculate net input."""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray | int:
        """Return class label after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


def plot_decision_regions(X: np.ndarray, y: np.ndarray, classifier: Perceptron, resolution: float = 0.02) -> None:
    """Plot decision regions for a 2D dataset and a fitted classifier."""
    markers = ("o", "s", "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # class points
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f"Class {cl}",
            edgecolor="black",
        )
