import torch
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
        tqdm_silent=False   # ?
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
        pass 