import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from ignite.contrib.handlers import param_scheduler
import monai
from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    ScaleIntensityd,
)
from utils.logger import log
from utils.lr_schedulers import DiceCELoss


def _get_xforms(mode="train", keys=("image", "label"), img_size=(320, 320, 16)):
    """returns a composed transform."""

    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0),
                 mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityd(keys, minv=0.0, maxv=1.0),
    ]

    if mode in ["train"]:
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(img_size[0], img_size[1], -1),
                            mode="reflect"),  # ensure at least WxD
                RandAffined(
                    keys,
                    prob=0.25,
                    # 3 parameters control the transform on 3 dimensions
                    rotate_range=(0.1, 0.1, None),
                    scale_range=(0.2, 0.2, None),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1],
                                       spatial_size=img_size,
                                       num_samples=3),
                RandGaussianNoised(keys[0], prob=0.20, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )

        dtype = (np.float32, np.uint8)

    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)

    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])

    return monai.transforms.Compose(xforms)


def get_dataloader(cfg, mode, keys, data, img_size):
    if mode == 'train':
        transforms = _get_xforms("train", keys, img_size)
    elif mode == 'val':
        transforms = _get_xforms("val", keys, img_size)
    else:
        # Test
        transforms = _get_xforms("infer", keys, img_size)

    dataset = monai.data.CacheDataset(
        data=data,
        transform=transforms
    )

    return monai.data.DataLoader(
        dataset,
        # if == 1 ==> image-level batch to the sliding window method, not the window-level batch
        batch_size=cfg.batch_size,
        shuffle=cfg.loader.shuffle,
        num_workers=cfg.loader.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_model(cfg):
    """returns a unet model instance."""
    try:
        return getattr(monai.networks.nets, cfg.model.name)(**cfg.model.params)
    except:
        log(f"Failed to load model. Model: {cfg.model.name}")


def get_loss(cfg):
    if cfg.loss.name == "DiceCELoss":
        return DiceCELoss()

    try:
        return getattr(monai.losses, cfg.loss.name)(**cfg.loss.params)
    except:
        log(
            f"Failed to import and load the loss function. Loss Function {cfg.loss.name}"
        )


def get_optimizer(cfg, parameters):
    optimizer = getattr(torch.optim, cfg.optimizer.name)(
        parameters, **cfg.optimizer.params)

    log(f'optim: {cfg.optimizer.name}')

    return optimizer


def get_scheduler(cfg, optimizer, len_loader):
    try:
        if cfg.scheduler.name == "CosineAnnealingScheduler":
            return getattr(param_scheduler, cfg.scheduler.name)(
                optimizer, cycle_size=len_loader, **cfg.scheduler.params)
        else:
            return getattr(param_scheduler, cfg.scheduler.name)(
                optimizer, **cfg.scheduler.params)
    except:
        log(f"Failed to load the scheduler. Scheduler: {cfg.scheduler.name}")


def get_inferer(patch_size):
    """returns a sliding window inference instance."""

    sw_batch_size, overlap = 2, 0.5
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",

    )
    return inferer
