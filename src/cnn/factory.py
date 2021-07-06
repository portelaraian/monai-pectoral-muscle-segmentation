import glob

import numpy as np

import torch
from ignite.contrib.handlers import param_scheduler
from ignite.contrib.handlers import ProgressBar

import monai
from monai.transforms import (
    CastToTyped,
    ToTensord
)

from utils.logger import log


def _get_transforms(transforms, dtype=(np.float32, np.uint8), keys=("image", "label")):
    """Returns a composed transform.

    Args:
        mode (str, optional): Mode speficied (e.g. train/test). Defaults to "train".
        img_size (tuple, optional): Spatial image size. Defaults to (320, 320, 16).

    Returns:
        [monai.transforms]: Returns MONAI transforms composed.
    """
    def get_object(transform):
        if hasattr(monai.transforms, transform.name):
            return getattr(monai.transforms, transform.name)(**transform.params)
        else:
            return eval(transform.name)

    xforms = [get_object(transform) for transform in transforms]

    xforms.extend(
        [
            CastToTyped(keys=keys, dtype=dtype),
            ToTensord(keys=keys),
        ]
    )

    return monai.transforms.Compose(xforms)


def get_dataloader(cfg, data):
    """Apply the transforms and create a DataLoader.

    Args:
        cfg (config file): Config file from model.
        data (list): List containing all the files (in this case the MRIs).

    Returns:
        monai.data.DataLoader: Returns a DataLoader.
    """
    transforms = _get_transforms(cfg.transforms)

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


def get_post_transforms(post_transforms=None):
    return monai.transforms.Compose([
        monai.transforms.AsDiscreted(keys=("pred", "label"),
                                     argmax=(True, False),
                                     to_onehot=True,
                                     n_classes=2)
    ])


def get_model(cfg):
    """Instantiates the model.

    Args:
        cfg (config file): Config file from model.

    Returns:
        Pytorch (MONAI) model: Returns a model instance.
    """
    if cfg.model.name == "DynUNet":
        raise ValueError(f"Not supporting {cfg.model.name} anymore.")

    try:
        return getattr(monai.networks.nets, cfg.model.name)(**cfg.model.params)
    except:
        log(f"Failed to load model. Model: {cfg.model.name}")


def get_loss(cfg):
    """Instantiate the loss function.

    Args:
        cfg (config file): Config file from model.

    Returns:
        monai.losses: Returns a monai instance loss.
    """
    log(f"Criterion: {cfg.loss.name}")

    try:
        return getattr(monai.losses, cfg.loss.name)(**cfg.loss.params)
    except:
        log(
            f"Failed to import and load the loss function. Loss Function {cfg.loss.name}"
        )


def get_optimizer(cfg, parameters):
    """Get the optimizer.

    Args:
        cfg (config file): Config file from model.
        parameters (model.params): Params from the model.

    Returns:
        torch.optim: Returns a optimizer (Pytorch).
    """
    optimizer = getattr(torch.optim, cfg.optimizer.name)(
        parameters, **cfg.optimizer.params)

    log(f"Optimizer: {cfg.optimizer.name}")
    return optimizer


def get_scheduler(cfg, optimizer, len_loader):
    """Get scheduler.

    Args:
        cfg (config file): Config file from model.
        optimizer (torch.optim): Optimizer.
        len_loader (int): Len of the DataLoader.

    Returns:
        lr_scheduler (ignite): Returns a learning rate scheduler.
    """
    log(f"LR Scheduler: {cfg.scheduler.name}")

    try:
        if cfg.scheduler.name == "CosineAnnealingScheduler":
            return getattr(param_scheduler, cfg.scheduler.name)(
                optimizer, cycle_size=len_loader, **cfg.scheduler.params)
        else:
            return getattr(
                torch.optim.lr_scheduler,
                cfg.scheduler.name
            )(optimizer, **cfg.scheduler.params)
    except:
        log(f"Failed to load the scheduler. Scheduler: {cfg.scheduler.name}")


def get_inferer(cfg):
    """Returns a sliding window inference instance

    Args:
        cfg (Config file): cfg (config file): Config file..

    Returns:
        monai.inferer: Returns a MONAI inferer.
    """
    try:
        return getattr(monai.inferers, cfg.inferer.name)(**cfg.inferer.params)
    except:
        log(
            f"Failed to import and load the loss function. Loss Function {cfg.inferer.name}"
        )


def get_handlers(cfg, handler, model=None, fold=None, evaluator=None, scheduler=None):
    def get_object(handler):
        if hasattr(monai.handlers, handler.name):
            return getattr(monai.handlers, handler.name)
        else:
            return eval(handler.name)

    handlers = [get_object(_handler)(**_handler.params)
                for _handler in handler.handlers]

    if handler.name == "validation":
        handlers.extend([
            monai.handlers.CheckpointSaver(
                save_dir=cfg.workdir,
                file_prefix=f"model_fold{fold}",
                save_dict={
                    "model": model
                },
                save_key_metric=True,
                key_metric_n_saved=5)
        ])
    else:
        handlers.extend([
            monai.handlers.ValidationHandler(
                validator=evaluator,
                interval=5,
                epoch_level=True

            ),
            monai.handlers.LrScheduleHandler(
                lr_scheduler=scheduler, print_lr=True,)
        ])

    return handlers


def get_models(cfg, device):

    if type(cfg.checkpoints) != list:
        model_paths = glob.glob(cfg.checkpoints)
        models = []

    for model_path in model_paths:
        model = get_model(cfg).to(device)
        model.load_state_dict(torch.load(model_path))
        models.append(model)

    log(f"Total models successfully loaded: {len(models)}")

    return models
