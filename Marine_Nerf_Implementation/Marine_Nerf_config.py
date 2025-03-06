from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig

from Marine_Nerf_Implementation.Marine_Nerf_Model import Marine_Nerf_ModelConfig
from Marine_Nerf_Implementation.Marine_Nerf_dataparser import Marine_Nerf_Dataparser_Config
from Marine_Nerf_Implementation.Marine_Nerf_datamanager import Marine_Nerf_DataManager, Marine_Nerf_DataManagerConfig
from Marine_Nerf_Implementation.Marine_Nerf_pipeline import Marine_Nerf_Pipeline, Marine_Nerf_PipelineConfig


Marine_Nerf = MethodSpecification(
  config=TrainerConfig(
    method_name="Marine-Nerf",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=Marine_Nerf_PipelineConfig(
        datamanager=Marine_Nerf_DataManagerConfig(
            _target=Marine_Nerf_DataManager[DepthDataset],
            pixel_sampler=PairPixelSamplerConfig(),
            dataparser=Marine_Nerf_Dataparser_Config(),
            train_num_rays_per_batch=15000,
            eval_num_rays_per_batch=4096,
        ),
        model=Marine_Nerf_ModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
  ),
  description="Custom description"
)
