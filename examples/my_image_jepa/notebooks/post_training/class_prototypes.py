"""Helpers for the clustering_features notebook.

Per-class "prototype" representations — the mean backbone feature per CIFAR-10 class —
and the class-to-class distance matrices (Euclidean & cosine) between them.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from dataset import get_val_transforms

# CIFAR-10 label order
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def compute_class_centroids(model, data_root="./data", batch_size=256,
                            device=None, verbose=True):
    """Mean backbone feature per CIFAR-10 class (the class 'prototype' / centroid).

    For each class, averages the D-dim backbone features over ALL its training images.
    Uses the ORIGINAL images (val transform = ToTensor + Normalize only, no views /
    augmentation), so the prototype reflects the class, not an augmentation. Averaging
    over the ~5000 images per class cancels per-image accidents (pose, background,
    lighting) and keeps what is common to the class.

    Args:
        model:      the encoder; called as model(x) -> (features, projection).
        data_root:  CIFAR-10 root dir (expects it already downloaded).
        batch_size: scan batch size.
        device:     torch device; defaults to mps if available else cpu.
        verbose:    print device, per-class image counts, and the centroid shape.

    Returns:
        centroids: [n_classes, feat_dim] numpy array — one mean feature vector per class.
    """
    n_classes = len(CIFAR10_CLASSES)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if verbose:
        print(f"device: {device}")

    # val transform = ToTensor + Normalize only (the original image, not an augmented view)
    class_ds = CIFAR10(root=data_root, train=True, download=False,
                       transform=get_val_transforms())
    class_loader = DataLoader(class_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    sums = None                          # [n_classes, feat_dim], allocated on first batch
    counts = torch.zeros(n_classes)

    was_training = model.training
    model.eval()
    model.to(device)
    with torch.no_grad():
        for x, y in class_loader:
            feats, _ = model(x.to(device))          # [B, feat_dim] backbone features
            feats = feats.cpu()
            if sums is None:
                sums = torch.zeros(n_classes, feats.shape[1])
            for c in range(n_classes):
                m = (y == c)
                if m.any():
                    sums[c] += feats[m].sum(0)
                    counts[c] += m.sum()
    model.to("cpu")
    if was_training:
        model.train()

    centroids = (sums / counts[:, None]).numpy()    # [n_classes, feat_dim] mean per class

    if verbose:
        print("images per class:", dict(zip(CIFAR10_CLASSES, counts.int().tolist())))
        print("centroid matrix:", centroids.shape)

    return centroids


def plot_class_distance_matrices(centroids, classes=None, verbose=True):
    """Class-to-class distance heatmaps between the class prototypes.

    Draws two matrices side by side:
      - Euclidean : ||c_i - c_j||         (magnitude of the raw difference vector)
      - Cosine    : 1 - cos(c_i, c_j)     (direction only; ignores overall magnitude)

    Each uses a diverging scale centered on the TYPICAL (median) off-diagonal distance,
    so blue = closer-than-typical and red = farther-than-typical, giving full contrast
    on both sides. The diagonal (self-distance) is dropped from the scale and shown gray.

    Args:
        centroids: [n_classes, feat_dim] class prototypes (from compute_class_centroids).
        classes:   class names for the axis labels; defaults to CIFAR10_CLASSES.
        verbose:   print a few specific class-pair comparisons.

    Returns:
        (euc, cos): the two [n_classes, n_classes] distance matrices.
    """
    if classes is None:
        classes = CIFAR10_CLASSES
    n_classes = len(classes)

    # euclidean: magnitude of the raw difference vector between two class prototypes
    euc = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)   # [n, n]
    # cosine distance: 1 - cos(angle); ignores overall magnitude, compares direction only
    unit = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    cos = 1.0 - unit @ unit.T

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    for ax, D, title in [(axes[0], euc, "Euclidean distance  ||c_i - c_j||"),
                         (axes[1], cos, "Cosine distance  1 - cos(c_i, c_j)")]:
        Dm = D.copy()
        np.fill_diagonal(Dm, np.nan)                        # drop self-distances from the scale
        off = Dm[~np.isnan(Dm)]
        vmin, vmid, vmax = off.min(), float(np.median(off)), off.max()
        vmid = min(max(vmid, vmin + 1e-9), vmax - 1e-9)     # keep strictly between for TwoSlopeNorm
        # diverging scale centered on the TYPICAL distance -> full contrast on both sides
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)

        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad("#eeeeee")                             # diagonal (NaN) shown as light gray
        im = ax.imshow(Dm, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(n_classes)); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(n_classes)); ax.set_yticklabels(classes)
        ax.set_title(f"{title}\nblue = closer than median ({vmid:.2f}),  red = farther", fontsize=10, loc="left")
        for i in range(n_classes):
            for j in range(n_classes):
                if i == j:
                    continue
                t = float(norm(D[i, j]))                    # 0..1 position on the color scale
                ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (t < 0.22 or t > 0.78) else "#222222")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("class-to-class distance between mean encoder representations (original images)", fontsize=13)
    plt.show()

    if verbose:
        def dist(a, b, D):
            return D[classes.index(a), classes.index(b)]

        for name, D in [("Euclidean", euc), ("Cosine", cos)]:
            fh = dist("frog", "horse", D)
            fc = dist("frog", "automobile", D)   # "car"
            cs = dist("automobile", "ship", D)   # "car" vs "boat"
            print(f"\n--- {name} ---")
            print(f"d(frog, horse)       = {fh:.3f}")
            print(f"d(frog, car)         = {fc:.3f}")
            print(f"  frog closer to horse than to car?  {fh < fc}   (gap {fc - fh:+.3f})")
            print(f"d(frog, horse)       = {fh:.3f}   vs   d(car, boat) = {cs:.3f}   (|diff| {abs(fh - cs):.3f})")

    return euc, cos
