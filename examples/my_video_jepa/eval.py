import collections 

import numpy as np 
import torch 
import torch.nn.functional as F 
import wandb
from einops import rearrange, repeat
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def add_label_to_video(video, label): 
    """Add a text label overlay on each frame of a video
    
    Args: 
        video: numpy array of shape (T, H, H, C) in uint8
        label: text string to add 
    
    Returns:
        numpy array of shape (T, H, W, C)
    """
    font = ImageFont.load_default()
    T, H, W, C = video.shape 

    labeled_frames = []
    for t in range(T): 
        frame = Image.fromarray(video[t])
        draw = ImageDraw(frame, "RGBA")
        draw.rectangle([0, 0, W, 20], fill =(40, 40, 40, 200))
        draw.text((4,4), label, fill=(255, 255, 255), font=font)
        labeled_frames.append(np.array(frame))
    return np.stack(labeled_frames, axis = 0)

def visualize_videos(
        batch,
        jepa,
        pixel_decoder,
        detection_head,
        num_samples
):
    """Create visualization videos for wandb logging.
    
    Returns a list of videos, each with 3 vertically stacked rows:
    1. Ground Truth video 
    2. Predicted rollout reconstruction 
    3. Digit detection overlay
    """

    x = batch["video"]
    x_jepa = jepa.encoder(x)

    T = x.shape[2]
    preds, _ = jepa.unroll(
        x,
        actions=None,
        nsteps= T - 2,
        unroll_mode = "parallel",
        compute_loss=False,
        return_all_steps=True
    )

    # One step predictions
    one_step_pred = x_jepa[:, :, 1:].clone()

