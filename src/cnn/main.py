import os
import sys
import argparse
import glob

import torch

import monai
from monai.handlers import (
    SegmentationSaver,
    MeanDice,
    StatsHandler,
    HausdorffDistance,
    StatsHandler,
    MetricsSaver,

)
from monai.transforms import (
    AsDiscreted,
    MeanEnsembled
)

from utils.config import Config
from utils.logger import logger, log
from utils.util import SplitDataset


import factory
sys.path.append(os.path.abspath(os.path.join("./")))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    """Parse the arguments.

    Returns:
        parser: parser containing the parameters.
    """
    parser = argparse.ArgumentParser(
        description="Runs the segmentation algorithm.")

    parser.add_argument(
        "mode",
        metavar="mode",
        choices=("train", "test"),
        type=str,
        help="mode of workflow"
    )
    parser.add_argument("config")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fold", type=int)

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
    cfg.fold = args.fold

    logger.setup(cfg.workdir, name='%s_model_%s_config' %
                 (cfg.mode, cfg.model.name))

    torch.cuda.set_device(cfg.gpu)

    monai.config.print_config()
    monai.utils.set_determinism(seed=cfg.seed)

    log(f"Using: {DEVICE}")
    log(f"Model: {cfg.model.name}")
    log(f"Mode: {cfg.mode}")

    if cfg.mode == "train":
        train(cfg)

    elif cfg.mode == "test":
        test(cfg)

    else:
        raise ValueError("Unknown mode.")


def train(cfg):
    """Run a training pipeline.

    Args:
        cfg (dict): config file.
    """

    data = sorted(glob.glob(os.path.join(
        cfg.data.train.imgdir, "mri/*.nii.gz")))

    log(f"Training: image/label ({len(data)}) folder: {cfg.data.train.imgdir}")

    # split dataset
    dataset_splitted = SplitDataset(data, cfg.seed)
    train_files, val_files = dataset_splitted.get_data(
        current_fold=cfg.fold,
        keys=cfg.keysd,
        path_to_masks_dir=os.path.join(cfg.data.valid.imgdir, "masks")
    )

    batch_size = cfg.batch_size
    log(f"Batch size: {batch_size}")

    # creating data loaders
    train_loader = factory.get_dataloader(cfg.data.train, train_files)
    val_loader = factory.get_dataloader(cfg.data.valid, val_files)

    model = factory.get_model(cfg).to(DEVICE)
    optimizer = factory.get_optimizer(cfg, model.parameters())
    scheduler = factory.get_scheduler(cfg, optimizer, len(train_loader))
    criterion = factory.get_loss(cfg)
    inferer = factory.get_inferer(cfg)

    run_nn(
        cfg=cfg,
        current_fold=cfg.fold,
        model=model,
        inferer=inferer,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader
    )


def run_nn(cfg, current_fold, model, inferer, optimizer, scheduler, criterion, train_loader, val_loader):
    """Run the deep cnn model

    Args:
        cfg (dict): config file.
        current_fold (int): current fold to train the model.
        model (monai.networks.nets): MONAI architecture to be trained.
        inferer (monai.inferers): sliding window method for model inference.
        optimizer (torch.optim): pytorch optimizer.
        scheduler (torch.optim.lr_scheduler): pytorch lr scheduler.
        criterion (monai.losses): MONAI loss function.
        train_loader (dataloader): train dataloader.
        val_loader (dataloader): validation dataloader.
    """

    log("")
    log("#"*33)
    log(f"## Started to train on fold: {current_fold} ##")
    log("#"*33)

    val_handlers = factory.get_handlers(
        cfg,
        cfg.handlers.val,
        model,
        current_fold
    )

    post_transforms = factory.get_post_transforms(
        cfg.data.train.post_transforms
    )

    # create evaluator (to be used to measure model quality during training)
    evaluator = monai.engines.SupervisedEvaluator(
        device=DEVICE,
        val_data_loader=val_loader,
        network=model,
        inferer=inferer,
        post_transform=post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=False,
                output_transform=lambda x: (x["pred"], x["label"]),
            )
        },
        val_handlers=val_handlers,
        amp=cfg.amp,
    )

    train_handlers = factory.get_handlers(
        cfg,
        cfg.handlers.train,
        model,
        current_fold,
        evaluator,
        scheduler
    )

    trainer = monai.engines.SupervisedTrainer(
        device=DEVICE,
        max_epochs=cfg.epochs,
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=criterion,
        inferer=inferer,
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=cfg.amp,
    )

    trainer.run()


def test(cfg):
    """Perform evalutaion and save the segmentations.

     Args:
        cfg (dict): config file.
    """
    images = sorted(
        glob.glob(os.path.join(cfg.data.test.imgdir, "mri/*.nii.gz"))
    )
    labels = sorted(
        glob.glob(os.path.join(cfg.data.test.imgdir, "masks/*.nii"))
    )

    log(f"Testing: image/label ({len(images)}/{len(labels)}) folder: {cfg.data.test.imgdir}")

    test_files = [
        {"image": img, "label": seg}
        for img, seg in zip(images, labels)
    ]

    val_loader = factory.get_dataloader(cfg.data.test, test_files)
    models = factory.get_models(cfg, DEVICE)

    pred_keys = [f"pred{idx}" for idx in range(len(models))]

    mean_post_transforms = monai.transforms.Compose(
        [
            MeanEnsembled(
                keys=pred_keys,
                output_key="pred",
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
        models,
        pred_keys,
    )


def ensemble_evaluate(cfg, post_transforms, loader, models, pred_keys):
    """Ensemble method for evaluation.

    Args:
        cfg (config file): Config file from model.
        post_transforms (transforms): MONAI transforms.
        loader (DataLoader): torch test DataLoader.
        models (list): list of models with its respective checkpoints.
        pred_keys (list): list containing pred keys (e.g. ["pred1", "pred2", ..., "predn"]).
    """
    evaluator = monai.engines.EnsembleEvaluator(
        device=DEVICE,
        val_data_loader=loader,
        pred_keys=pred_keys,
        networks=models,
        inferer=factory.get_inferer(cfg),
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
        amp=cfg.amp,
    )

    StatsHandler(
        name="evaluator",
        # no need to print loss value, so disable per iteration output
        output_transform=lambda x: None,
    ).attach(evaluator)

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
