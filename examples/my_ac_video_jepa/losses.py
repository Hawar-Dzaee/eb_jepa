import torch 
import torch.nn as nn 
import torch.nn.functional as F 


def sq_loss(x, y, reduction="mean"):
    """Simple square loss (MSE)."""
    return nn.functional.mse_loss(x, y, reduction=reduction)

def square_cost_seq(state, predi):
    """Square loss between two [B, C, T, H, W] sequences."""
    return sq_loss(state, predi)

class SquareLossSeq(nn.Module):
    """Square loss over a sequence [B, C, T, H, W] (feature dim at dim 1)."""

    def __init__(self, proj=None):
        super().__init__()
        self.proj = nn.Identity() if proj is None else proj 

    def forward(self, state, predi):
        state = self.proj(state.transpose(0, 1).flatten(1).transpose(0, 1))
        predi = self.proj(predi.transpose(0, 1).flattne(1).transpose(0, 1))
        return square_cost_seq(state, predi)

class HingeStdLoss(torch.nn.Module):
    def __init__(
        self,
        std_margin: float = 1.0,
    ):
        """
        Encourages each feature to maintain at least a minimum standard deviation. 
        Features with std below the margin incur a penalty of (std_margin -std). 
        Args:
            std_margin (float, default=1.0)
                Minimum desired standard deviation per feature. 
        """
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            std_loss: Scalar tensor with the hinge loss on standard deviation 
        """
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss
    

class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelation.

        Normalizes by D * (D -1) where D is feature dimensionality. 
        """
        super().__init__()
    
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m 
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def forward(self, x: torch.Tensor):
        """
        Args: 
            x: [N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        # Calcualte off-diagonal loss 
        cov_loss = self.off_diagonal(cov).pow(2).mean()

        return cov_loss
    
class TemporalSimilarityLoss(torch.nn.Module):
    def __init__(self):
        """
        Temporal Similarity Loss. 
        Encourages consecutive frames to have similar representations by penalizing 
        the squared didfference between consecutive time steps. 
        """
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, N, D] where T is time steps, N is batch size, D is feature dimension
        """
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device)
        sim_loss_t = (x[1:] - x[:-1]).pow(2).mean()
        return sim_loss_t
    

class InverseDynamicLOss(torch.nn.Module):
    def __init__(self, idm: nn.Module):
        """
        Predicts actions from consecutive states and compares with ground truth actions. 
        Args:
            Idm (nn.Module): Inverse dynamics model that takes (state_t, state_t+1)
        """
        super().__init__()
        self.idm = idm 

    def forward(self, x:torch.Tensor, actions: torch.Tensor):
        """
        Args:
            x: [T, B, D] - States across time steps 
            actions: [B, A, T] - Ground truth actions between consecutive states 
        """
        if x.shape[0] <= 1 or actions is None:
            return torch.tensor(0.0, device=x.device)
        
        t, b, d = x.shape

        states_t = x[:-1].transpose(0, 1) # [B, T-1, D]
        states_t_plus_1 = x[1:].transpose(0, 1) # [B, T-1, D]

        states_t_flat = states_t.reshape(-1, d) # [B*(T-1), D]
        states_t_plus_1_flat = states_t_plus_1.reshape(-1, d) #[B*(T-1), D]

        pred_actions = self.idm(states_t_flat, states_t_plus_1_flat) # [B*(T-1), A]
        target_actions = actions.transpose(1, 2)[: , :-1].reshape(
            -1, actions.size(1)
        )   # [B*(T-1), A]
        idm_loss = F.mse_loss(pred_actions, target_actions)

        return idm_loss

class VC_IDM_Sim_Regularizer(torch.nn.Module):
    def __init__(
        self, 
        cov_coeff: float,
        std_coeff: float,
        sim_coeff_t: float,
        idm_coeff: float = 0.0,
        idm: nn.Module = None,
        std_margin: float = 1,
        first_t_only: bool = True,
        projector: nn.Module = None,
        spatial_as_samples: bool = False,
        sim_t_after_proj: bool = False,
        idm_after_proj: bool = False
    ):
        """
        Composite Regularizer combining multiple losses 

        This is a composite loss that combines:
        - Hinge Standard Deviation Loss 
        - Covariance Decorrelation Loss 
        - Temporal Similarity Loss 
        - Inverse Dynamic Model Loss
        
        Args:
            cov_coeff (float): Weight for covariance loss 
            std_coeff (float): Weight for std hinge loss 
            sim_coeff_t (float): Weight for temporal similarity loss 
            idm_coeff (float): Weight for inverse dynamics loss 
            idm (nn.Module): Inverse dynamics model
            std_margin (float): Minimum desired std per feature 
            first_t_only (bool): Use only first time slice for std/cov loss 
            projector (nn.Module): Optional projection layer
            spatial_as_samples (bool): Treat spatial location as samples
            sim_t_after_proj (bool): Apply temporal loss after projection
            idm_after_proj (bool): Apply IDM loss after projection 
        """
        super().__init__()
        self.cov_coeff = cov_coeff
        self.std_coeff = std_coeff
        self.sim_coeff_t = sim_coeff_t
        self.idm_coeff = idm_coeff

        self.first_t_only = first_t_only
        self.projector = nn.Identity() if projector is None else projector
        self.spatial_as_samples = spatial_as_samples
        self.idm_after_proj = idm_after_proj

        # Initialize individual loss components 
        self.std_loss_fn = HingeStdLoss(std_margin=std_margin)
        self.cov_loss_fn = CovarianceLoss()
        self.sim_loss_fn = TemporalSimilarityLoss()
        self.idm_loss_fn 