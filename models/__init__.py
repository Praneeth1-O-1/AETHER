"""AETHER model components — public API.

Import structure::

    from models import AETHERModel
    from models import OpticalEncoder, SAREncoder, DEMEncoder
    from models import CrossModalAlphaFusion
    from models import Decoder
    from models import LULCHead, get_task_head, register_task_head
"""

from models.optical_encoder import OpticalEncoder
from models.sar_encoder import SAREncoder
from models.dem_encoder import DEMEncoder
from models.crossmodal_fusion import CrossModalAlphaFusion
from models.decoder import Decoder
from models.task_heads import (
    TaskHead,
    LULCHead,
    get_task_head,
    register_task_head,
    list_task_heads,
)
from models.aether import AETHERModel

__all__ = [
    "AETHERModel",
    "OpticalEncoder",
    "SAREncoder",
    "DEMEncoder",
    "CrossModalAlphaFusion",
    "Decoder",
    "TaskHead",
    "LULCHead",
    "get_task_head",
    "register_task_head",
    "list_task_heads",
]
