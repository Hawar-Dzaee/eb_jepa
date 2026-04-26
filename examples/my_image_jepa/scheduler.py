import torch 

class WarmupCosineScheduler: 
    """Warmup cosine learning rate scheduler"""

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        base_lr,
        min_lr = 0.0,
        warmup_start_lr = 3e-5
    ):

        self.optimizer = optimizer 
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr 


    def step(self,epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_start_lr + epoch * (
                self.base_lr - self.warmup_start_lr
            ) / (self.warmup_epochs - 1)

        else : 
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 
                + torch.cos(
                    torch.tensor(
                        (epoch - self.warmup_epochs)
                        / (self.max_epochs - self.warmup_epochs)
                        * 3.14159
                    )
                )
            )

        for param_group in self.optimizer.param_groups: 
            param_group["lr"] = lr 