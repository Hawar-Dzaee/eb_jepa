
from model import ResNet18,ImageSSL
from lars import LARS
from scheduler import WarmupCosineScheduler


def run(
        fnmae: str = "examples/my_image_jepa/default.yaml",
        cfg = None,
        folder = None,
        **overrides
):
    pass 
