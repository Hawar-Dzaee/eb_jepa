from pathlib import Path 


import wandb
from omegaconf import OmegaConf

from model import ResNet18,ImageSSL
from lars import LARS
from scheduler import WarmupCosineScheduler
from log_utils import get_logger
from training_utils import (
    load_config,
    setup_device,
    setup_seed,
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    setup_wandb
)


logger = get_logger(__name__)

def run(
        fnmae: str = "examples/my_image_jepa/default.yaml",
        cfg = None,
        folder = None,
        **overrides
):
    """
    Train an Image JEPA (VICReg/BCS) model on CIFAR-10

    Args: 
        fname: Path to YAML config file 
        cfg: Pre-loaded config object (optional, overrides config file)
        folder: Experiment folder path (optional, auto-generated if not provided)
        **overrides: Config overrides in dot notation (e.g., optim.epochs=50)
    
    """ 

    # Load config 
    if cfg is None:
        cfg = load_config(fnmae,overrides if overrides else None)

    # Setup using shared utilities 
    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    # Create experiment directory using unified structure (if not provided)
    if folder is None:
        if cfg.meta.get("model_folder"):
            exp_dir = Path(cfg.meta.model_folder)
            folder_name = exp_dir.name 
            exp_name = folder_name.rsplit("_seed",1)[0]
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("image_jepa",cfg)
            exp_dir = get_unified_experiment_dir(
                example_name = "image_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed
            )

    else:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True,exist_ok=True)
        # Extract exp_name from folder name by removing _seed{seed} suffix
        folder_name = exp_dir.name  # e.g. "resnet_vicreg_seed1"
        exp_name = folder_name.rsplit("_seed", 1)[0] # e.g., "resnet_vicreg"

    wandb_run = setup_wandb(
        project= "eb_jepa",
        config={"exmaple": "image_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir= exp_dir,
        run_name=exp_name,
        tags=["image_jepa", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled= cfg.logging.log_wandb,
        sweep_id=cfg.logging.get("wandb_sweep_id")
    )

    logger.info("Loading CIFAR-10 dataset")
    
