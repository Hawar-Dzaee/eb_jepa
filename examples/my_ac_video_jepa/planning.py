import os 
import time 
from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from log_utils import get_logger
from vis_utils import(
    analyze_distances,
    create_comparison_gif,
    plot_losses,
    save_decoded_frames,
    save_gif,
    show_images
)

logger = get_logger(__name__)

planner_name_map = {
    "cem": "CEMPlanner",
    "mppi": "MPPIPlanner",
}
objective_name_app = {
    "repr_dist": "ReprTargetDistMPCObjective",
}

def main_unroll_eval(
    model,
    env_creator,
    eval_folder,
    num_samples=4,
    loader=None,
    prober=None,
    cfg=None,
):
    """
    Evaluate the model's capabilities by comparing unrolled predictions to ground truth.
    """
    env = env_creator()
    env.reset()
    device = next(model.parameters()).device
    normalizer = (
        loader.dataset.normalizer if hasattr(loader.dataset, "normalizer") else None
    )
    agent = GCAgent()


class GCAgent:
    def __init__(
        self,
        model,
        action_dim=2,
        plan_cfg=None, 
        normalizer: Optional[Callable] = None,
        loc_porber: Optional[Callable] = None, 
        img_prober: Optional[Callable] = None,
        env: Optional[Callable] = None 
    ):
        self.plan_cfg = plan_cfg
        self.env = env 
        self.model = model 
        self.device = next(model.parameters()).device
        self.loc_prober = loc_porber
        self.img_prober = img_prober
        self.normalizer = normalizer

        # Set default values if plan_cfg is None 
        if plan_cfg is None:
            self.decode_each_iteration = False 
            self.num_act_stepped = 1 
            self.planner = None 
            logger.info("No plan_cfg provided in GCAgent, planner not initialized.")
        else:
            self.decode_each_iteration = plan_cfg.planner.get(
                "decode_each_iteration", False 
            )
            self.num_act_stepped = plan_cfg.planner.get("num_act_stepped", 1)
            planner_name = plan_cfg.planner.get("planner_name", "cem")
            planner_class_name = planner_name_map[planner_name]
            planner_class = globals()[planner_class_name]
            if planner_class is not None:
                self.planner = planner_class(
                    unroll = self.unroll,
                    action_dim=action_dim,
                    decode_loc_to_pixel=self.decode_loc_to_pixel,
                    **plan_cfg.planner
                )
            else:
                logger.info("No planner provided in GCAgent.")
                self.planner = None 

        self.goal_state = None 
        self.goal_position = None 
        self.goal_state_enc = None 
        self._prev_losses = None 

    def decode_loc_to_pixel(self,predicted_encs, wall_x=None, door_y=None):
        """
        Decode the predicted encodings into frames. 

        Args:
            predicted_encs: [B, D, T, H, W]
        
        Returns:
            np.array of shape [B, T, H, W, C] on cpu for visualizaiton.
        """
        assert self.loc_prober is not None 
        B, D, T, H, W = predicted_encs.shape 
        out = self.loc_prober.apply_head(predicted_encs).permute(0, 2, 1).cpu() # B T 2 
        out = self.normalizer.unnormalize_location(out) # B T 2 
        frames = self.env_coord_to_pixel(out, wall_x=wall_x, door_y=door_y) # B T C H W
        frames = frames.permute(0, 1, 3, 4, 2).cpu().numpy() # B T H W C 
        return frames
    