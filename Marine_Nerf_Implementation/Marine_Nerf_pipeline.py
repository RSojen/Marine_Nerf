import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from Marine_Nerf_Implementation.Marine_Nerf_Model import Marine_Nerf_ModelConfig
from Marine_Nerf_Implementation.Marine_Nerf_datamanager import (
    Marine_Nerf_DataManager,
    Marine_Nerf_DataManagerConfig,
)


@dataclass
class Marine_Nerf_PipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: Marine_Nerf_Pipeline)
    """target class to instantiate"""
    datamanager: Marine_Nerf_DataManagerConfig = Marine_Nerf_DataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = Marine_Nerf_ModelConfig()
    """specifies the model config"""


class Marine_Nerf_Pipeline(VanillaPipeline):
    def __init__(
        self,
        config: Marine_Nerf_PipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        self.datamanager: Marine_Nerf_DataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank
        )
        self.datamanager.to(device)

        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size

