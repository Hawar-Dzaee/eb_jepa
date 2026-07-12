import copy 
import os 
from pathlib import Path 

import torch 
import torch.nn as nn 
import yaml
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast

from architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
    )

from eb_jepa.datasets.utils import init_data
from jepa import JEPA, JEPAProbe
from log_utils import get_logger
from losses import SquareLossSeq,VC_IDM_Sim_Regularizer
from state_decoder import MLPXYHead
from eb_jepa.training_utils import(
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_config,
    log_config,
    log_data_info,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb
)

logger = get_logger(__name__)

def run(
        fname: str = "cfgs/tarin.yaml",
        cfg=None,
        folder=None,
        **overrides,
):
    """
    Train an action-conditioned Video JEPA model.

    Args:
        fname: Path to the YAML config file. 
        cfg: Pre-loaded config object (optinoal, overrides config file).
        folder: Experiment folder path (optional, auto-generated if not provided).
        **overrides: Config overrides in dot notation (e.g., model.henc=64)
    """ 

    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Create experiment directory using unified structure (if not provide)
    if folder is None:
        if cfg.meta.get("model_folder"):
            folder = Path(cfg.meta.model_folder)
            folder_name = folder.name 
            exp_name = folder_name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("ac_video_jepa",cfg)
            folder = get_unified_experiment_dir(
                example_name="ac_video_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed
            )
    else:
        folder = Path(folder)
        folder_name = folder.name
        exp_name = folder_name.rsplit("_seed", 1)[0]

    os.makedirs(folder, exists_ok=True)

    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
    )

    # --SETUP 
    setup_device("auto")
    setup_seed(cfg.meta.seed)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    # --WANDB
    wandb_run = setup_wandb(
        project="eb_jepa",
        config={
            "example":"ac_video_jepa",
            **OmegaConf.to_container()
        },
        run_dir=folder,
        run_name=exp_name,
        tags=[f"seed_{cfg.meta.seed}", "ac_video_jepa"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb",False),
        sweep_id=cfg.logging.get("wandb_sweep_id")
    )

    log_data_info(
        cfg.data.env_name,
        len(loader),
        data_config.batch_size,
        train_samples=data_config.size,
        val_samples=data_config.val_size
    )

    # Mixed Precision setup
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)
    use_amp = cfg.training.get("use_amp", True)
    scaler = GradScaler(device.type, enabled=use_amp)
    logger.info(f"Using AMP with {dtype}" if use_amp else f"AMP disabled")

    # -- ENV (for plan/unroll eval)
    enable_eval = cfg.meta.get("enable_plan_eval", False)
    env_creator = None 
    plan_cfg = None 
    num_eval_episodes = 10 

    if enable_eval:
        if cfg.meta.eval_every_itr <= 0:
            cfg.meta.eval_every_itr = len(loader)
        with open(cfg.eval.plan_cfg_path, "r") as f:
            plan_cfg = yaml.load(f, Loader=yaml.FullLoader)
        plan_cfg["logging"] = copy.deepcopy(dict(cfg.logging))
        with open(cfg.eval.eval_cfg_path,"r") as f:
            eval_cfg_dict = yaml.safe_load(f)
        _, _, env_config = init_data(
            env_name=cfg.data.env_name, cfg_data=dict(eval_cfg_dict.get("data", {}))
        )
        num_eval_episodes = eval_cfg_dict.get("meta", {}).get("num_eval_episoded",10)

        def env_creator():
            from eb_jepa.datasets.two_rooms.env import DotWall

            cfg_eval_env = eval_cfg_dict.get("env")
            return DotWall(
                config=env_config,
                **cfg_eval_env
            )
        
    # -- SAVE CONFIG
    latest_ckpt = folder / "latest.pth.tar"
    steps_per_epoch = data_config.size // data_config.batch_size
    total_steps = cfg.optim.epochs * steps_per_epoch
    config_path = folder / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, config_path)
    print(f"Saved complete config to {config_path}")

    # --MODEL 
    test_input = torch.rand(
        (
            1,
            cfg.model.dobs,
            1,
            data_config.img_size,
            data_config.img_size
        )
    )
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=cfg.model.dobs,
        final_ln=True,
        mlp_outputdim=512,
        input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size),
    )
    test_output = encoder(test_input)
    _, f, _, h, w = test_output.shape
    predictor = RNNPredictor(
        hidden_size=encoder.mlp_outputdim, final_ln=encoder.final_ln
    )
    aencoder = nn.Identity()
    if cfg.model.regulaizer.use_proj:
        projector = Projector(
            f"{encoder.mlp_outputdim}-{encoder.mlp_outputdim*4}-{encoder.mlp_outputdim*4}"
        )
    else:
        projector = None 
    logger.info(f"Encoder output: {tuple(test_output.shape)}")
    idm = InverseDynamicsModel(
        state_dim=h
        * w
        * (projector.out_dim if cfg.model.regularizer.idm_after_proj else f),
        hidden_dim=256,
        action_dim=2
    ).to(device)

    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg.model.regularizer.cov_coeff,
        std_coeff=cfg.model.regularizer.std_coeff,
        sim_coeff_t=cfg.model.regularizer.sim_coeff_t,
        idm_coeff= cfg.model.regularizer.get("idm_coeff", 0.1),
        idm = idm,
        first_t_only=cfg.model.regularizer.get("first_t_only"),
        projector=projector,
        spatial_as_samples=cfg.model.regularizer.spatial_as_samples,
        idm_after_proj=cfg.model.regularizer.idm_after_proj,
        sim_t_after_proj=cfg.model.regularizer.sim_t_after_proj
    )
    ploss = SquareLossSeq()
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss).to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    log_model_info(jepa, {"encoder": encoder_params, "predictor": predictor_params})

    log_config(cfg)

    #--PROBER
    xy_head = MLPXYHead()