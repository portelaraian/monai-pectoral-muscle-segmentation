import torch
import numpy as np
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from monai.engines import SupervisedEvaluator
from ignite.utils import to_onehot
from monai.transforms import AsDiscrete, Activations
from monai.metrics import compute_hausdorff_distance, compute_meandice, compute_average_surface_distance
from monai.handlers import (
    CheckpointSaver,
    CheckpointLoader,
    SegmentationSaver,
    MeanDice,
    StatsHandler,
    ValidationHandler,
    HausdorffDistance,
)
from monai.transforms import (
    RandGaussianNoised
)
import pandas as pd
import numpy as np
import monai
from utils.config import Config
import glob
from utils.logger import logger, log
import os
import sys
import argparse
import factory
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join("./")))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    """Parse the arguments

    Returns:
        parser: parser containing the parameters.
    """
    parser = argparse.ArgumentParser(
        description="Runs the segmentation algorithm.")

    parser.add_argument("mode", metavar="mode", default="train",
                        choices=("train", "test", "test-segment"),
                        type=str, help="mode of workflow"
                        )
    parser.add_argument("config")
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


def main():
    """Set the main configurations and run the mode specified.

    Raises:
        ValueError: Unknown mode specified.
    """
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
        model.load_state_dict(torch.load(cfg.trained_model_path))
        test_pytorch(cfg, model)
    else:
        raise ValueError("Unknown mode.")


def train(cfg, model):
    """Run a training pipeline.

    Args:
        cfg (config file): Config file from model.
        model (torch model): Pytorch MONAI model.
    """

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
    # _config_trainer_logging(cfg, trainer, cfg.model.name)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.run()


def test_ignite(cfg, model):
    """Perform evalutaion and save the segmentations

     Args:
        cfg (config file): Config file from model.
        model (torch model): Pytorch MONAI model.
    """
    images = sorted(glob.glob(
        os.path.join(cfg.data.test.imgdir, "mri/*.nii.gz")))
    labels = sorted(glob.glob(
        os.path.join(cfg.data.test.imgdir, "masks/*.nii")))

    log(f"Testing: image/label ({len(images)}/{len(labels)}) folder: {cfg.data.test.imgdir}")

    test_files = [{"image": img, "label": seg}
                  for img, seg in zip(images, labels)]
    keys = ("image", "label")

    # creating data loaders
    val_loader = factory.get_dataloader(
        cfg.data.test, 'val',
        keys, test_files, cfg.imgsize
    )

    # create evaluator (to be used to measure model quality during training)
    val_post_transforms = monai.transforms.Compose([
        AsDiscreted(keys=("pred", "label"),
                    argmax=(True, False),
                    to_onehot=True,
                    n_classes=2)
    ])

    val_handlers = [
        ProgressBar(),
        StatsHandler(name="evaluator", output_transform=lambda x: None),
        CheckpointLoader(load_path=cfg.trained_model_path,
                         load_dict={"model": model}),
        SegmentationSaver(
            output_dir=cfg.prediction_folder,
            output_ext=".nii",
            batch_transform=lambda batch: batch["image_meta_dict"],
            output_transform=lambda output: output["pred"],
        ),
    ]

    evaluator = SupervisedEvaluator(
        device=DEVICE,
        val_data_loader=val_loader,
        network=model,
        inferer=factory.get_inferer(cfg.imgsize),
        post_transform=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=lambda x: (x["pred"], x["label"])),
        },
        additional_metrics={
            "val_hausdorff_distance": HausdorffDistance(include_background=True, output_transform=lambda x: (x["pred"], x["label"])),
        },
        val_handlers=val_handlers,
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
        amp=cfg.amp,
    )

    evaluator.run()


def test_pytorch(cfg, model):
    """Perform evalutaion and save the segmentations

     Args:
        cfg (config file): Config file from model.
        model (torch model): Pytorch MONAI model.
    """
    images = sorted(glob.glob(
        os.path.join(cfg.data.test.imgdir, "mri/*.nii.gz")))
    labels = sorted(glob.glob(
        os.path.join(cfg.data.test.imgdir, "masks/*.nii")))

    log(f"Testing: image/label ({len(images)}/{len(labels)}) folder: {cfg.data.test.imgdir}")

    test_files = [{"image": img, "label": seg}
                  for img, seg in zip(images, labels)]
    keys = ("image", "label")

    # creating data loader
    val_loader = factory.get_dataloader(
        cfg.data.test, 'val',
        keys, test_files, cfg.imgsize
    )

    inferer = factory.get_inferer(cfg.imgsize)
    saver = monai.data.NiftiSaver(
        output_dir=cfg.prediction_folder,
        output_ext=".nii",
        mode="nearest"
    )

    results = {
        "id": [],
        "spatial_size": [],
        "dice": [],
        "hausdorff_dist": [],
        "avg_surface_dist": []
    }

    with torch.no_grad():
        for infer_data in tqdm(val_loader):

            # ID
            results["id"].append(
                infer_data["label_meta_dict"]["filename_or_obj"][0]
            )

            # Spatial size of the current tensor
            results["spatial_size"].append(
                infer_data["label_meta_dict"]["spatial_shape"][0].cpu().numpy()
            )

            preds = inferer(infer_data[keys[0]].to(DEVICE), model)
            labels = infer_data[keys[1]].to(DEVICE)

            n = 1.0
            for _ in range(4):
                # TTA
                _img = RandGaussianNoised(
                    keys[0],
                    prob=1.0,
                    std=0.01
                )(infer_data)[keys[0]]

                pred = inferer(_img.to(DEVICE), model)
                preds = preds + pred
                n = n + 1.0

                for dims in [[2], [3]]:
                    flip_pred = inferer(
                        torch.flip(_img.to(DEVICE), dims=dims),
                        model
                    )
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0

            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()

            # Dice
            results["dice"].append(
                compute_meandice(
                    y_pred=preds,
                    y=labels
                ).cpu().numpy()[0][0])

            # Hausdorff distance
            results["hausdorff_dist"].append(
                compute_hausdorff_distance(
                    y_pred=preds,
                    y=labels,
                    include_background=True
                ).cpu().numpy()[0][0]
            )

            # Average surface distance
            results["avg_surface_dist"].append(
                compute_average_surface_distance(
                    y_pred=preds,
                    y=labels,
                    include_background=True
                ).cpu().numpy()[0][0]
            )

            # Save predictions (segmentations:  format)
            # saver.save_batch(preds, infer_data["image_meta_dict"])

    results = pd.DataFrame(results)
    print(results)

    print(f"Mean-dice: {np.mean(results['dice'])}")
    print(f"Mean-hausdorff: {np.mean(results['hausdorff'])}")
    print(f"Average Surface Distance: {np.mean(results['ASD'])}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    log(torch.backends.cudnn.benchmark)

    torch.cuda.empty_cache()

    try:
        main()
    except KeyboardInterrupt:
        log('Keyboard Interrupted')
