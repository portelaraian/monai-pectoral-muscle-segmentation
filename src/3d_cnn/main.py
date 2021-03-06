import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.utils import setup_logger
from monai.transforms import AsDiscreted
from monai.metrics import DiceMetric, compute_roc_auc
from monai.handlers import (
    CheckpointSaver,
    MeanDice,
    StatsHandler,
    ValidationHandler,
    HausdorffDistance,
    ROCAUC,
    ConfusionMatrix
)
from monai.transforms import (
    RandGaussianNoised
)
import monai
from utils.config import Config
import glob
from utils.logger import logger, log
import os
import shutil
import sys
import argparse
import factory
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join("./")))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(
        description="Runs the segmentation algorithm.")

    parser.add_argument("mode", metavar="mode", default="train",
                        choices=("train", "test"),
                        type=str, help="mode of workflow"
                        )
    parser.add_argument("config")
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


def main():
    args = get_args()
    cfg = Config.fromfile(args.config)

    cfg.mode = args.mode
    cfg.gpu = args.gpu

    logger.setup(cfg.workdir, name='%s_model_%s_config' %
                 (cfg.mode, cfg.model.name))

    torch.cuda.set_device(cfg.gpu)

    monai.config.print_config()
    monai.utils.set_determinism(seed=cfg.seed)

    model = factory.get_model(cfg).to(DEVICE)

    log(f"Model: {cfg.model.name}")

    if cfg.mode == "train":
        log(f"Mode: {cfg.mode}")
        train(cfg, model)
    elif cfg.mode == "test":
        log(f"Mode: {cfg.mode}")
        infer(cfg, model)
    else:
        raise ValueError("Unknown mode.")


def _config_trainer_logging(trainer, name=None):
    trainer.logger = setup_logger(
        name=name,
        level=10,  # Debug level
        filepath=os.path.join(cfg.workdir, "trainerIgnite_%s.log" % (name))
    )


def train(cfg, model):
    """run a training pipeline."""

    images = sorted(glob.glob(
        os.path.join(cfg.data.train.imgdir, "mri/*.nii.gz")))
    labels = sorted(glob.glob(
        os.path.join(cfg.data.train.imgdir, "masks/*.nii")))

    log(f"Training: image/label ({len(images)}) folder: {cfg.data.train.imgdir}")

    keys = ("image", "label")
    train_frac, val_frac = cfg.train_frac, cfg.val_frac
    n_train = int(train_frac * len(images)) + 1
    n_val = min(len(images) - n_train, int(val_frac * len(images)))

    log(f"Training: train {n_train} val {n_val}, folder: {cfg.data.train.imgdir}")

    train_files = [{keys[0]: img, keys[1]: seg}
                   for img, seg in zip(images[:n_train], labels[:n_train])]
    val_files = [{keys[0]: img, keys[1]: seg}
                 for img, seg in zip(images[-n_val:], labels[-n_val:])]

    # create a training data loader
    batch_size = cfg.batch_size
    log(f"Batch size: {batch_size}")

    # creating data loaders
    train_loader = factory.get_dataloader(
        cfg.data.train, cfg.mode,
        keys, train_files, cfg.imgsize
    )

    val_loader = factory.get_dataloader(
        cfg.data.valid, 'val',
        keys, val_files, cfg.imgsize
    )

    optimizer = factory.get_optimizer(cfg, model.parameters())
    scheduler = factory.get_scheduler(cfg, optimizer, len(train_loader))
    criterion = factory.get_loss(cfg)

    log(f"Optimizer: {cfg.optimizer.name}")
    log(f"LR Scheduler: {cfg.scheduler.name}")
    log(f"Criterion: {cfg.loss.name}")

    # create evaluator (to be used to measure model quality during training)
    val_post_transform = monai.transforms.Compose([
        AsDiscreted(keys=("pred", "label"),
                    argmax=(True, False),
                    to_onehot=True,
                    n_classes=2)
    ])

    val_handlers = [
        ProgressBar(),
        CheckpointSaver(save_dir=cfg.workdir,
                        save_dict={"model": model},
                        save_key_metric=True,
                        key_metric_n_saved=20),
    ]

    evaluator = monai.engines.SupervisedEvaluator(
        device=DEVICE,
        val_data_loader=val_loader,
        network=model,
        inferer=factory.get_inferer(cfg.imgsize),
        post_transform=val_post_transform,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=False, output_transform=lambda x: (x["pred"], x["label"])),
        },
        val_handlers=val_handlers,
        amp=cfg.amp,
    )

    # evaluator as an event handler of the trainer
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        StatsHandler(tag_name="train_loss",
                     output_transform=lambda x: x["loss"]),
    ]

    trainer = monai.engines.SupervisedTrainer(
        device=DEVICE,
        max_epochs=cfg.epochs,
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=criterion,
        inferer=factory.get_inferer(cfg.imgsize),
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=cfg.amp,
    )
    _config_trainer_logging(trainer, cfg.model.name)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.run()


def infer(cfg, model):
    """
    run inference, the output folder will be "./output"
    """
    try:
        checkpoints = sorted(glob.glob(cfg.checkpoints))
        log("loaded %s checkpoints" % (len(checkpoints)))
    except Exception as e:
        log(f"{e} : unable to load checkpoints")
        return

    checkpoint = checkpoints[-1]

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    log("-"*10)
    log(f"using {checkpoint}.")

    images = sorted(glob.glob(os.path.join(cfg.data.test.imgdir, "*.nii.gz")))
    log(f"infer: image ({len(images)}) folder: {cfg.data.test.imgdir}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)

    # creating data loaders
    infer_loader = factory.get_dataloader(
        cfg.data.test, cfg.mode,
        keys, infer_files, cfg.imgsize
    )

    inferer = factory.get_inferer(cfg.imgsize)
    saver = monai.data.NiftiSaver(
        output_dir=cfg.prediction_folder,
        mode="nearest"
    )

    with torch.no_grad():
        for infer_data in tqdm(infer_loader, total=len(infer_loader)):
            preds = inferer(infer_data[keys[0]].to(DEVICE), model)
            n = 1.0
            for _ in range(4):
                # test time augmentations
                _img = RandGaussianNoised(keys[0],
                                          prob=1.0,
                                          std=0.01)(infer_data)[keys[0]]
                pred = inferer(_img.to(DEVICE), model)
                preds = preds + pred
                n = n + 1.0
                for dims in [[2], [3]]:
                    flip_pred = inferer(torch.flip(_img.to(DEVICE),
                                                   dims=dims), model)
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0

            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()
            saver.save_batch(preds, infer_data["image_meta_dict"])


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    log(torch.backends.cudnn.benchmark)

    torch.cuda.empty_cache()

    try:
        main()
    except KeyboardInterrupt:
        log('Keyboard Interrupted')
