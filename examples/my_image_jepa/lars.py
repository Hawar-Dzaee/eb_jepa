import torch 
import torch.optim as optim
from torch.optim.optimizer import required




class LARS(optim.Optimizer): 
    def __init__(
        self,
        params,
        lr = required, 
        momentum = 0, 
        dampening = 0, 
        weight_decay = 0, 
        nesterov = False,
        eta = 1e-3,
        eps = 1e-8,
        clip_lr = False,
        exclude_bias_n_norm = False, 
    ):
        if lr is not required and lr < 0.0 : 
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0 : 
            raise ValueError(f"Invalid momentum values: {momentum}")
        if weight_decay < 0.0 : 
            raise ValueError(f"Invalid weight_decay values: {weight_decay}")

        defaults = dict(
            lr = lr, 
            momentum = momentum,
            dampening = dampening,
            weight_decay = weight_decay,
            nesterov = nesterov, 
            eta = eta, 
            eps = eps,
            clip_lr = clip_lr, 
            exclude_bias_n_norm = exclude_bias_n_norm
        )

        if nesterov and (momentum <= 0 or dampening != 0): 
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params,defaults)     

    def __setstate__(self,state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("Nesterov",False)

    
    @torch.no_grad()
    def step(self,closure = None):
        """Performs a single optimization step. 
        Args : 
            closure (callable, optional): A closure that reevaluates the model 
                and returns the loss 
        """
        loss = None 
        if closure is not None:
            with torch.enable_grad(): 
                loss = closure()

        # exlude scaling for params with 0 weight decay 
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None : 
                    continue 

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part 
                if p.ndim != 1 or not group["exclude_bias_n_norm"]: # it doesn't make sense to use LARS on ndim == 1, 
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (
                            g_norm + p_norm * weight_decay + group["eps"]
                        )
                        lars_lr *= group["eta"]

                        # clip lr 
                        if group["clip_lr"]: 
                            lars_lr = min(lars_lr / group["lr"],1)
                        
                        d_p = d_p.add(p, alpha = weight_decay)  # d_p​ ← d_p ​ + λ⋅p
                        d_p *= lars_lr

                # sgd part 
                if momentum != 0 : 
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else : 
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha= 1 - dampening)
                    if nesterov : 
                        d_p = d_p.add(buf,alpha = momentum)
                    else : 
                        d_p = buf 

                p.add_(d_p,alpha=-group["lr"])

        return loss 
