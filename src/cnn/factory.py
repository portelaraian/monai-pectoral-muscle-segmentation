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
        transforms (list): list containing all transforms specified on config file (cfg).
        dtype (tuple, optional): dtypes used on CastToTyped MONAI transform. Defaults to (np.float32, np.uint8).
        keys (tuple, optional): keys used as params for MONAI transforms . Defaults to ("image", "label").

    Returns:
        monai.transforms: returns MONAI transforms composed.
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
        cfg (dict): config file.
        data (list): list containing all the files (in this case the MRIs).

    Returns:
        monai.data.DataLoader: returns a DataLoader.
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


def get_post_transforms(transforms):
    """Returns MONAI post transforms composed.

    Args:
        transforms (dict): python dict containing all transforms and its parameters.
    """
    def get_object(post_transform):
        if hasattr(monai.transforms, post_transform.name):
            return getattr(monai.transforms, post_transform.name)(**post_transform.params)
        else:
            return eval(post_transform.name)

    post_transforms = [get_object(post_transform)
                       for post_transform in transforms]

    return monai.transforms.Compose(post_transforms)


def get_model(cfg):
    """Instantiates the model.

    Args:
        cfg (dict): config file.

    Returns:
        Pytorch (MONAI) model: returns a model instance.
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
        cfg (dict): config file.

    Returns:
        monai.losses: returns a monai instance loss.
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
        cfg (dict): config file.
        parameters (model.params): params from the model.

    Returns:
        torch.optim: returns a optimizer (pytorch).
    """
    optimizer = getattr(torch.optim, cfg.optimizer.name)(
        parameters, **cfg.optimizer.params)

    log(f"Optimizer: {cfg.optimizer.name}")
    return optimizer


def get_scheduler(cfg, optimizer, len_loader):
    """Get scheduler.

    Args:
        cfg (dict): config file.
        optimizer (torch.optim): optimizer.
        len_loader (int): len of the DataLoader.

    Returns:
        lr_scheduler (ignite): returns a learning rate scheduler.
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
        cfg (dict): config file.

    Returns:
        monai.inferer: returns a MONAI inferer.
    """
    try:
        return getattr(monai.inferers, cfg.inferer.name)(**cfg.inferer.params)
    except:
        log(
            f"Failed to import and load the loss function. Loss Function {cfg.inferer.name}"
        )


def get_handlers(cfg, handler, model=None, fold=None, evaluator=None, scheduler=None):
    """Returns the handlers specified on config file (cfg).

    Args:
        cfg (dict): config file.
        handler (list): list of all handlers and its parameters.
        model (monai.networks.nets, optional): architecture used for training. Defaults to None.
        fold (int, optional): current fold. Defaults to None.
        evaluator (monai.engines.SupervisedEvaluator, optional): evaluator used for validation. Defaults to None.
        scheduler (torch.optim.lr_scheduler, optional): lr scheduler used for training. Defaults to None.

    Returns:
        handlers (list): list containing all handlers loaded.
    """
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
    """Load models for testing.

    Args:
        cfg (dict): config file.
        device (str): device used. (eg.:. 'cpu' or 'cuda')

    Returns:
        list: return all the models loaded.
    """

    if type(cfg.checkpoints) != list:
        model_paths = glob.glob(cfg.checkpoints)
        models = []

    for model_path in model_paths:
        model = get_model(cfg).to(device)
        model.load_state_dict(torch.load(model_path))
        models.append(model)

    log(f"Total models successfully loaded: {len(models)}")

    return models
