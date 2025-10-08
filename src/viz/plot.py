from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3D projection

# --- Visual style helpers -----------------------------------------------------


def _apply_clean_style(ax: plt.Axes) -> None:
    """Make axes white with subtle gridlines and crisper spines/ticks."""
    ax.set_facecolor("white")
    ax.grid(
        True, which="major", color="#666666", alpha=0.3, linestyle="-", linewidth=0.5
    )
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    ax.tick_params(colors="#333333")


def _default_label_map(y: np.ndarray, labels: tuple[str, str] | None) -> dict:
    """Return a mapping from unique y values to human labels, with Link first."""
    classes = np.unique(y)
    if labels is not None:
        if len(labels) < len(classes):
            raise ValueError("Not enough labels provided for the number of classes.")
        return {cl: labels[i] for i, cl in enumerate(classes)}
    if set(classes.tolist()) == {0, 1}:
        return {1: "Link", 0: "Non-Link"}
    return {cl: f"Class {cl}" for cl in classes}


def _clean_feature_names(
    feature_names: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    """Replace *_sim with *_score if present."""
    if feature_names is None:
        return None
    return tuple(name.replace("_sim", "_score") for name in feature_names)


# Accessibility-friendly palette
_MARKERS = ("o", "s", "^", "v", "<")
_POINT_COLORS = ("#e68600", "#1f77b4")  # orange, blue
_REGION_COLORS = ("#ffd6a5", "#cfe1ff", "#dff3df", "#e8dcf3", "#f1e2d5")


# --- 2D scatter ---------------------------------------------------------------


def plot_data_scatter_2d(
    X: np.ndarray,
    y: np.ndarray,
    labels: tuple[str, str] | None = None,
    feature_names: tuple[str, str] | None = None,
) -> None:
    """Scatter plot of 2D features with class labels."""
    label_map = _default_label_map(y, labels)
    feature_names = _clean_feature_names(feature_names)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    _apply_clean_style(ax)

    classes = sorted(np.unique(y), key=lambda cl: 0 if label_map[cl] == "Link" else 1)

    for idx, cl in enumerate(classes):
        ax.scatter(
            X[y == cl, 0],
            X[y == cl, 1],
            color=_POINT_COLORS[idx % len(_POINT_COLORS)],
            marker=_MARKERS[idx % len(_MARKERS)],
            label=label_map[cl],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
        )

    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.02),
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )
    for text in leg.get_texts():
        text.set_color("#222222")

    fig.tight_layout()
    plt.show()


# --- Learning curve -----------------------------------------------------------


def plot_learning_curve(errors: list[int]) -> None:
    """Plot convergence: number of updates per epoch."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    _apply_clean_style(ax)

    ax.plot(
        range(1, len(errors) + 1), errors, marker="o", linewidth=2.0, color="#1f77b4"
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Number of updates")

    fig.tight_layout()
    plt.show()


# --- 2D decision regions ------------------------------------------------------


def plot_decision_regions_2d(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    feat_idx: tuple[int, int] = (0, 1),
    resolution: float = 0.02,
    feature_names: tuple[str, str] | None = None,
    labels: tuple[str, str] | None = None,
) -> None:
    """Plot decision regions using any 2 selected features from X."""
    label_map = _default_label_map(y, labels)
    feature_names = _clean_feature_names(feature_names)
    X_plot = X[:, feat_idx]

    classes = np.unique(y)
    region_cmap = ListedColormap(_REGION_COLORS[: len(classes)])

    pad = 0.05
    x1_min, x1_max = X_plot[:, 0].min() - pad, X_plot[:, 0].max() + pad
    x2_min, x2_max = X_plot[:, 1].min() - pad, X_plot[:, 1].max() + pad
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )

    grid2 = np.c_[xx1.ravel(), xx2.ravel()]
    grid_full = np.zeros((grid2.shape[0], X.shape[1]))
    grid_full[:, feat_idx[0]] = grid2[:, 0]
    grid_full[:, feat_idx[1]] = grid2[:, 1]

    Z = classifier.predict(grid_full).reshape(xx1.shape)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    _apply_clean_style(ax)

    ax.contourf(xx1, xx2, Z, alpha=0.45, cmap=region_cmap, antialiased=True)

    try:
        if set(np.unique(Z).tolist()) == {0, 1}:
            ax.contour(xx1, xx2, Z, levels=[0.5], colors="#222222", linewidths=1.0)
        else:
            ax.contour(
                xx1,
                xx2,
                Z,
                levels=np.unique(Z),
                colors="#222222",
                linewidths=0.6,
                linestyles="--",
            )
    except Exception:
        pass

    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    classes = sorted(classes, key=lambda cl: 0 if label_map[cl] == "Link" else 1)

    for idx, cl in enumerate(classes):
        ax.scatter(
            X_plot[y == cl, 0],
            X_plot[y == cl, 1],
            alpha=0.95,
            c=_POINT_COLORS[idx % len(_POINT_COLORS)],
            marker=_MARKERS[idx % len(_MARKERS)],
            label=label_map[cl],
            edgecolor="white",
            linewidth=0.8,
        )

    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.02),
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )
    for text in leg.get_texts():
        text.set_color("#222222")

    fig.tight_layout()
    plt.show()


# --- 3D decision plane --------------------------------------------------------


def plot_decision_plane_3d(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    feat_idx: tuple[int, int, int] = (0, 1, 2),
    feature_names: tuple[str, str, str] | None = None,
    labels: tuple[str, str] | None = None,
) -> None:
    """Plot the perceptron decision boundary as a plane in 3D (using 3 features)."""
    label_map = _default_label_map(y, labels)
    feature_names = _clean_feature_names(feature_names)
    Xp = X[:, feat_idx]

    fig = plt.figure(figsize=(5, 4.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_facecolor("white")
    ax.xaxis._axinfo["grid"].update(
        color="#666666", linestyle="-", linewidth=0.5, alpha=0.3
    )
    ax.yaxis._axinfo["grid"].update(
        color="#666666", linestyle="-", linewidth=0.5, alpha=0.3
    )
    ax.zaxis._axinfo["grid"].update(
        color="#666666", linestyle="-", linewidth=0.5, alpha=0.3
    )

    classes = np.unique(y)
    classes = sorted(classes, key=lambda cl: 0 if label_map[cl] == "Link" else 1)

    for idx, cl in enumerate(classes):
        ax.scatter(
            Xp[y == cl, 0],
            Xp[y == cl, 1],
            Xp[y == cl, 2],
            alpha=0.9,
            label=label_map[cl],
            color=_POINT_COLORS[idx % len(_POINT_COLORS)],
            edgecolor="white",
            linewidth=0.6,
            s=28,
        )

    w = np.asarray(classifier.w_)[list(feat_idx)]
    b = float(classifier.b_)
    x_range = np.linspace(Xp[:, 0].min(), Xp[:, 0].max(), 25)
    y_range = np.linspace(Xp[:, 1].min(), Xp[:, 1].max(), 25)
    xx, yy = np.meshgrid(x_range, y_range)
    if w[2] != 0:
        zz = (-b - w[0] * xx - w[1] * yy) / w[2]
        ax.plot_surface(
            xx,
            yy,
            zz,
            alpha=0.25,
            color="#d8b4f8",  # pale purple
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
        )

    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])

    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.02),
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )
    for text in leg.get_texts():
        text.set_color("#222222")

    fig.tight_layout()
    plt.show()
