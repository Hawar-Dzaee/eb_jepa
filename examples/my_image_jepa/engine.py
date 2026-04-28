import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

def train_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        linear_probe,
        scaler,
        device,
        epoch,
        loss_fn,
        use_amp=True,
        dtype=torch.float16,
        tqdm_silent=False   
):
    """Train for one epoch."""
    model.train()
    linear_probe.train()

    # Dynamic loss accumulator 
    loss_totals = {}
    total_linear_loss = 0 
    linear_correct = 0 
    linear_total = 0 

    pbar = tqdm(train_loader,desc=f"Epoch {epoch}",disable=tqdm_silent)
    for batch_idx,(views,target) in enumerate(pbar):
        view1,view2 = views[0].to(device,non_blocking=True),views[1].to(device,non_blocking=True)
        target = target.to(device, non_blocking = True)

        with autocast(device.type, enabled=use_amp, dtype=dtype):
            features, z1 = model(view1) 
            _, z2 = model(view2) 
            loss_dict = loss_fn(z1,z2)
            loss = loss_dict["loss"]

        with torch.no_grad():
            features_frozen = features.detach().float()

        linear_outputs = linear_probe(features_frozen)
        linear_loss = F.cross_entropy(linear_outputs,target) 

        _, predicted = linear_outputs.max(1)
        linear_correct_batch = predicted.eq(target).sum().item()

        total_loss_batch = loss + linear_loss

        optimizer.zero_grad()
        scaler.scale(total_loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics dynamically based on loss_dict keys 
        for key,value in loss_dict.items():
            if key not in loss_totals:
                loss_totals[key] = 0 
            loss_totals[key] += value.item()
        total_linear_loss += linear_loss.item()

        # Update linear probe accuracy (pre-computed under autocast)
        linear_total += target.size(0)
        linear_correct += linear_correct_batch

        # Update progress bar 
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Linear": f"{linear_loss.item():.4f}",
                "Acc": f"{100.*linear_correct/linear_total:.2f}%",
            }
        )

    # Update learning rate
    scheduler.step(epoch)

    # Build return dict dynamically 
    num_batches = len(train_loader)
    metrics = {key: total / num_batches for key, total in loss_totals.items()}
    metrics["linear_loss"] = total_linear_loss / num_batches
    metrics["linear_acc"] = 100.0 * linear_correct / linear_total

    return metrics


