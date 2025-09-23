from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3D projection


def plot_data_scatter_2d(
    X: np.ndarray,
    y: np.ndarray,
    labels: tuple[str, str] | None = None,
    feature_names: tuple[str, str] | None = None,
) -> None:
    """Scatter plot of 2D features with class labels (e.g. Setosa vs Versicolor)."""
    colors = ("red", "blue")
    markers = ("o", "s")
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            X[y == cl, 0],
            X[y == cl, 1],
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            label=labels[idx] if labels else f"Class {cl}",
            alpha=0.8,
            edgecolor="black",
        )
    if feature_names:
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(errors: list[int]) -> None:
    """Plot convergence: number of updates per epoch."""
    plt.plot(range(1, len(errors) + 1), errors, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Number of updates")
    plt.tight_layout()
    plt.show()


def plot_decision_regions_2d(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    feat_idx: tuple[int, int] = (0, 1),
    resolution: float = 0.02,
    feature_names: tuple[str, str] | None = None,
) -> None:
    """Plot decision regions using any 2 selected features from X."""
    X_plot = X[:, feat_idx]
    markers = ("o", "s", "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    x1_min, x1_max = X_plot[:, 0].min() - 0.05, X_plot[:, 0].max() + 0.05
    x2_min, x2_max = X_plot[:, 1].min() - 0.05, X_plot[:, 1].max() + 0.05
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )

    # build a full feature grid (zeros for non-plotted dims)
    grid2 = np.c_[xx1.ravel(), xx2.ravel()]
    grid_full = np.zeros((grid2.shape[0], X.shape[1]))
    grid_full[:, feat_idx[0]] = grid2[:, 0]
    grid_full[:, feat_idx[1]] = grid2[:, 1]

    Z = classifier.predict(grid_full).reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            X_plot[y == cl, 0],
            X_plot[y == cl, 1],
            alpha=0.8,
            c=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            label=f"Class {cl}",
            edgecolor="black",
        )

    if feature_names:
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_decision_plane_3d(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    feat_idx: tuple[int, int, int] = (0, 1, 2),
    feature_names: tuple[str, str, str] | None = None,
) -> None:
    """Plot the perceptron decision boundary as a plane in 3D (using 3 features)."""
    Xp = X[:, feat_idx]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # scatter points
    for cl in np.unique(y):  # avoid Ruff B007
        ax.scatter(
            Xp[y == cl, 0],
            Xp[y == cl, 1],
            Xp[y == cl, 2],
            alpha=0.7,
            label=f"Class {cl}",
        )

    # plane: w·x + b = 0  ->  z = (-b - w0*x - w1*y) / w2
    w = np.asarray(classifier.w_)[list(feat_idx)]
    b = float(classifier.b_)
    x_range = np.linspace(Xp[:, 0].min(), Xp[:, 0].max(), 20)
    y_range = np.linspace(Xp[:, 1].min(), Xp[:, 1].max(), 20)
    xx, yy = np.meshgrid(x_range, y_range)
    if w[2] != 0:
        zz = (-b - w[0] * xx - w[1] * yy) / w[2]
        ax.plot_surface(xx, yy, zz, alpha=0.25)
    # else: vertical plane; skip surface

    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
