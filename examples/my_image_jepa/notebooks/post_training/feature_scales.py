"""Helpers for the understanding_features notebook.

Dataset-global color scales for the activation heatmaps. Computing them once, over
the whole train set, lets every heatmap in the notebook share the same ceilings so a
given color means the same magnitude everywhere.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

# A three-hue colormap (low -> high): blue -> magenta -> gold. More perceptual
# resolution than a single-hue Blues ramp, but few enough hues to stay clean when the
# data clusters low — the bulk reads as blue->magenta instead of a muddy violet wash.
RICH_ANCHORS = ["#3b6fe0", "#d6336c", "#f7b32b"]
RICH_CMAP = LinearSegmentedColormap.from_list("blue_magenta_gold", RICH_ANCHORS)


def compute_global_color_scales(dataset, model, batch_size=256, keep_every=4,
                                device=None, verbose=True):
    """Dataset-global color scales (p99) for the activation heatmaps.

    Scans the dataset once and returns three ceilings, computed as the 99th
    percentile (not the max) so a handful of outlier neurons don't wash out the
    heatmap — the distribution has a heavy tail (max ~13 but p99 ~0.9), so scaling
    to max wastes 90%+ of the color range.

      global_vmax  -> activation ceiling
      global_dmax  -> |view1 - view2|            (across-VIEW difference)
      global_sdmax -> |sample_i - sample_j|      (across-SAMPLE difference, first view)

    Args:
        dataset:    a dataset yielding ((view1, view2), label); scanned with shuffle=False
                    so the roll-by-1 across-sample trick uses real neighbors.
        model:      the encoder; called as model(x) -> (features, projection).
        batch_size: scan batch size.
        keep_every: keep every Nth batch to bound memory; p99 is robust to subsampling.
        device:     torch device; defaults to mps if available else cpu.
        verbose:    print the resulting ceilings.

    Returns:
        (global_vmax, global_dmax, global_sdmax, global_diffmax) as floats, where
        global_diffmax = max(global_dmax, global_sdmax) is one shared ceiling for BOTH
        difference columns, so a color means the same magnitude whether it's an
        across-view or an across-sample difference.
    """
    scan_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if verbose:
        print(f"device: {device}")

    act_samples = []     # subsample of activations
    diff_samples = []    # subsample of |view1 - view2|          (across-view)
    sdiff_samples = []   # subsample of |sample_i - sample_j|    (across-sample, first view)

    was_training = model.training
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, ((v1, v2), _) in enumerate(scan_loader):
            feat1, _ = model(v1.to(device))
            feat2, _ = model(v2.to(device))
            # keep every Nth batch to bound memory; p99 is robust to subsampling
            if i % keep_every == 0:
                act_samples.append(feat1.flatten().cpu())
                act_samples.append(feat2.flatten().cpu())
                diff_samples.append((feat1 - feat2).abs().flatten().cpu())
                # across-sample: each first-view feature minus the next sample's (roll by 1)
                sdiff_samples.append((feat1 - feat1.roll(1, 0)).abs().flatten().cpu())
    model.to("cpu")   # back to cpu so the earlier cells keep working as-is
    if was_training:
        model.train()

    act_samples = torch.cat(act_samples).numpy()
    diff_samples = torch.cat(diff_samples).numpy()
    sdiff_samples = torch.cat(sdiff_samples).numpy()

    global_vmax = np.quantile(act_samples, 0.99)
    global_dmax = np.quantile(diff_samples, 0.99)
    global_sdmax = np.quantile(sdiff_samples, 0.99)
    # one shared ceiling for BOTH difference columns, so a color means the same magnitude
    # whether it's an across-view or an across-sample difference. Use the larger of the two.
    global_diffmax = max(global_dmax, global_sdmax)

    if verbose:
        w = 46
        print(f"{'activation, single neuron':<{w}} | p99 : {global_vmax:.4f} | max : {act_samples.max():.4f}")
        print(f"{'across-view diff  |view1 - view2|, same image':<{w}} | p99 : {global_dmax:.4f} | max : {diff_samples.max():.4f}")
        print(f"{'across-sample diff  |this image - next image|':<{w}} | p99 : {global_sdmax:.4f} | max : {sdiff_samples.max():.4f}")

    return global_vmax, global_dmax, global_sdmax, global_diffmax


def plot_activation_distribution(dataset, model, n_batches=20, batch_size=256,
                                 device=None, verbose=True):
    """Plot the distribution of neuron activations over a sample of the dataset.

    Draws two histograms of all activation values (both views) — linear-y and log-y
    (the log axis exposes the heavy tail) — with the p99 marked as the heatmap color
    ceiling (global_vmax). This is the visual justification for clipping at p99
    instead of the max in the heatmaps.

    Args:
        dataset:    a dataset yielding ((view1, view2), label).
        model:      the encoder; called as model(x) -> (features, projection).
        n_batches:  number of batches to scan (20 x 256 x 2 x 512 ~ 5.2M values).
        batch_size: scan batch size.
        device:     torch device; defaults to mps if available else cpu.
        verbose:    print summary stats (zeros / mean / median / p99 / max).

    Returns:
        (acts, p99): the flattened activations (numpy) and the 99th percentile.
    """
    scan_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    acts = []
    was_training = model.training
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, ((v1, v2), _) in enumerate(scan_loader):
            feat1, _ = model(v1.to(device))
            feat2, _ = model(v2.to(device))
            acts.append(feat1.flatten().cpu())
            acts.append(feat2.flatten().cpu())
            if i + 1 >= n_batches:
                break
    model.to("cpu")
    if was_training:
        model.train()

    acts = torch.cat(acts).numpy()
    p99 = np.quantile(acts, 0.99)   # the heatmap color ceiling (global_vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].hist(acts, bins=100, color="#4C78A8")
    axes[0].set_title("all activations — linear y", fontsize=10, loc="left")
    axes[0].set_xlabel("activation")
    axes[0].set_ylabel("count")

    axes[1].hist(acts, bins=100, color="#4C78A8", log=True)
    axes[1].set_title("all activations — log y (shows the tail)", fontsize=10, loc="left")
    axes[1].set_xlabel("activation")
    axes[1].set_ylabel("count (log)")

    for ax in axes:
        ax.axvline(p99, color="#E45756", linestyle="--", linewidth=1.5, label=f"p99 = {p99:.3f}")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.show()

    if verbose:
        print(f"exact zeros: {(acts == 0).mean():.1%} of all values")
        print(f"mean: {acts.mean():.4f}, median: {np.median(acts):.4f}")
        print(f"p99: {p99:.4f}, max: {acts.max():.4f}")

    return acts, p99
