import torch
import numpy as np
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from monai.engines import SupervisedEvaluator
from ignite.utils import to_onehot
from ignite.engine.events import Events
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
    StatsHandler,
    MetricsSaver
)
from monai.transforms import (
    RandGaussianNoised,
    AsDiscreted,
    MeanEnsembled
)
import pandas as pd
import numpy as np
import monai
import glob
from utils.config import Config
from utils.logger import logger, log
from utils.util import SplitDataset
import os
import sys
from pathlib import Path
import argparse
import factory
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join("./")))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    """Parse the arguments.

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
    parser.add_argument('--snapshot')
    parser.add_argument('--output')

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
    cfg.snapshot = args.snapshot
    cfg.output = args.output

    logger.setup(cfg.workdir, name='%s_model_%s_config' %
                 (cfg.mode, cfg.model.name))

    torch.cuda.set_device(cfg.gpu)

    monai.config.print_config()
    monai.utils.set_determinism(seed=cfg.seed)

    log(f"Model: {cfg.model.name}")

    if cfg.mode == "train":
        log(f"Mode: {cfg.mode}")
        train(cfg)

    elif cfg.mode == "test":
        log(f"Mode: {cfg.mode}")
        test(cfg)

    else:
        raise ValueError("Unknown mode.")


def train(cfg):
    """Run a training pipeline.

    Args:
        cfg (config file): Config file from model.
    """

    data = sorted(glob.glob(os.path.join(
        cfg.data.train.imgdir, "mri/*.nii.gz")))

    log(f"Training: image/label ({len(data)}) folder: {cfg.data.train.imgdir}")

    keys = ("image", "label")

    # split dataset
    dataset_splitted = SplitDataset(data, cfg.seed)

    batch_size = cfg.batch_size
    log(f"Batch size: {batch_size}")

    num_models = 5
    models = [_run_nn(cfg, dataset_splitted, keys, idx)
              for idx in range(num_models)]


def _run_nn(cfg, dataset_splitted, keys, index):
    """Run the deep cnn model.

    Args:
        cfg (config file): Config file from model.
        dataset_splitted (list): list of the dataset splitted into n_folds.
        keys (tuple): dictionary keys used. E.g. ("image", "label").
        index (int): index indicating the fold index.

    Returns:
        torch model: returns a trained model 
    """
    log("")
    log("#"*33)
    log(f"## Started to train on fold: {index} ##")
    log("#"*33)

    train_files, val_files = dataset_splitted.get_data(
        current_fold=index,
        keys=keys,
        path_to_masks_dir=os.path.join(cfg.data.valid.imgdir, "masks"),
    )

    log(f"Train files: {len(train_files)} | Val files: {len(val_files)}")

    # creating data loaders
    train_loader = factory.get_dataloader(
        cfg.data.train, cfg.mode,
        keys, train_files, cfg.imgsize
    )

    val_loader = factory.get_dataloader(
        cfg.data.valid, 'val',
        keys, val_files, cfg.imgsize
    )

    model = factory.get_model(cfg).to(DEVICE)
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
                        file_prefix=f"{cfg.model_id}_fold{index}",
                        save_dict={"model": model},
                        save_key_metric=True,
                        key_metric_n_saved=5),
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

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.run()

    return model


def test(cfg):
    """Perform evalutaion and save the segmentations.

     Args:
        cfg (config file): Config file from model.
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

    if cfg.ensemble_evaluate:
        model_paths = glob.glob(cfg.checkpoints)
        models = []

        for model_path in model_paths:
            model = factory.get_model(cfg).to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            models.append(model)

        mean_post_transforms = monai.transforms.Compose(
            [
                MeanEnsembled(
                    keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
                    output_key="pred",
                    # in this particular example, we use validation metrics as weights
                    #weights=[0.95, 0.94, 0.95, 0.94, 0.90],
                ),
                AsDiscreted(keys=("pred", "label"),
                            argmax=(True, False),
                            to_onehot=True,
                            n_classes=2)
            ]
        )

        ensemble_evaluate(
            cfg,
            mean_post_transforms,
            val_loader,
            models
        )


def ensemble_evaluate(cfg, post_transforms, loader, models):
    """Ensemble method for evaluation.

    Args:
        cfg (config file): Config file from model.
        post_transforms (transforms): MONAI transforms.
        loader (DataLoader): torch test DataLoader.
        models ([list]): list of models with its respective checkpoints.
    """
    evaluator = monai.engines.EnsembleEvaluator(
        device=DEVICE,
        val_data_loader=loader,
        pred_keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
        networks=models,
        inferer=factory.get_inferer(cfg.imgsize),
        post_transform=post_transforms,
        key_val_metric={
            "test_mean_dice": MeanDice(
                include_background=False,
                output_transform=lambda x: (x["pred"], x["label"]),
            )
        },
        additional_metrics={
            "test_hausdorff": HausdorffDistance(
                include_background=False,
                output_transform=lambda x: (x["pred"], x["label"])
            )
        },
        amp=True,
    )

    val_stats_handler = StatsHandler(
        name="evaluator",
        # no need to print loss value, so disable per iteration output
        output_transform=lambda x: None,
    )
    val_stats_handler.attach(evaluator)

    # convert the necessary metadata from batch data
    SegmentationSaver(
        output_dir=cfg.prediction_folder,
        output_ext=".nii",
        output_postfix=cfg.model_id,
        name="evaluator",
        mode="nearest",
        batch_transform=lambda batch: batch["image_meta_dict"],
        output_transform=lambda output: output["pred"],
    ).attach(evaluator)

    MetricsSaver(
        save_dir=cfg.prediction_folder,
        delimiter=",",
        metric_details="*",
        batch_transform=lambda batch: batch["image_meta_dict"],

    ).attach(evaluator)

    evaluator.run()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    log(torch.backends.cudnn.benchmark)

    torch.cuda.empty_cache()

    try:
        main()
    except KeyboardInterrupt:
        log('Keyboard Interrupted')
