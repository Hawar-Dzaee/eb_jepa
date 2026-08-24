# Baseline acceptance criteria

*Pre-registered before the baseline run — written to fix the goalpost before the
result is known. Do not edit the criteria after seeing the run; record the outcome
in the annotated `baseline` git tag instead.*

## Reference
- Source: **reported** in `examples/image_jepa/README.md` (not re-run on this
  machine), "Impact of regularizations" table — batch 256, 1024×1024 projector,
  VICReg `std=1, cov=100`.
- Original val linear-probe accuracy = **90.12%**
  (the same config reads 90.05% in the projector table — ~0.07% run noise.)
- Caveat: because this is a reported number on the authors' hardware/seed, the
  tolerance band below must absorb hardware + library-version + seed differences,
  not seed noise alone.

## Reproduced if
- **Val linear-probe accuracy within ±2%** (absolute) of the reference number above.
  This is the real signal — measured at `main.py:273` via `evaluate_linear_probe`.
- **Loss curve converges** — no divergence, no plateau at a trivial value.
- **No collapse** — embedding std stays well above zero (VICReg's failure mode).

## Fixed setup
- Seed: **42** (`meta.seed` in `cfgs/default.yaml`)
- Config: `cfgs/default.yaml` (VICReg, projector on, 300 epochs)
- Downstream metric: val-set linear probe on frozen features

## Tolerance rationale
±2% covers hardware + library + seed differences against a reported number, while
staying tight enough to catch a real reimplementation bug: the README's own "good"
region is stable (~90.0–90.1% across logical hyperparameter choices), whereas
failure modes are catastrophic (std=100/cov=100 → 10% collapse), so ±2% cleanly
separates "reproduced" from "broken." Upgrade to a multi-seed noise estimate
(tolerance ≈ 2×std over 2–3 seeds) if any Phase-B result needs to be publishable —
or run the original here for an on-machine reference to tighten the band.

## Baseline frozen (2026-08-23) — pointers, not criteria
- **Result: PASS.** Val linear-probe **89.44%** (inside 88.12–92.12%). Curves confirm
  convergence (val_acc plateaus ~90, val_loss falls smoothly) and no collapse
  (`train_var_loss ≈ 0.48` at end). Run on a RunPod RTX 4090.
- **Code anchor:** `git tag baseline` → commit `00d6855` (pushed). Diff Phase-B work
  with `git diff baseline -- examples/my_image_jepa/`.
- **Metrics anchor:** wandb run `resnet_vicreg_proj_bs256_ep300_ph1024_po1024_std1.0_cov100.0`,
  tagged `baseline` in group `baseline`.
- **Weights + embeddings:** run dir
  `image_jepa/dev_2026-08-23_15-32/resnet_vicreg_proj_bs256_ep300_ph1024_po1024_std1.0_cov100.0_seed42/`
  holding `embedding_history.pt`, `latest.pth.tar`, `epoch_50…250.pth.tar`.
  Lives on the RunPod volume `/workspace/eb_jepa_data/checkpoints/…` **and** backed up
  locally at `AMI/eb_jepa_data/checkpoints/…` (sibling of the repo, outside git).
