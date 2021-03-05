import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
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
import monai
from utils.config import Config
import glob
import logging
import os
import shutil
import sys
import argparse
import factory
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

    torch.cuda.set_device(cfg.gpu)

    monai.config.print_config()
    monai.utils.set_determinism(seed=cfg.seed)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    model = factory.get_model(cfg).to(DEVICE)

    logging.info(f"Model: {cfg.model.name}")

    if cfg.mode == "train":
        logging.info(f"Mode: {cfg.mode}")
        train(cfg, model)
    elif cfg.mode == "test":
        logging.info(f"Mode: {cfg.mode}")
        infer(cfg, model)
    else:
        raise ValueError("Unknown mode.")


def train(cfg, model):
    """run a training pipeline."""

    images = sorted(glob.glob(
        os.path.join(cfg.data.train.imgdir, "mri/*.nii.gz")))
    labels = sorted(glob.glob(
        os.path.join(cfg.data.train.imgdir, "masks/*.nii")))

    logging.info(f"Training: image/label ({len(images)}) \
                 folder: {cfg.data.train.imgdir}"
                 )

    keys = ("image", "label")
    train_frac, val_frac = 0.8, 0.2
    n_train = int(train_frac * len(images)) + 1
    n_val = min(len(images) - n_train, int(val_frac * len(images)))

    logging.info(
        f"Training: train {n_train} val {n_val}, folder: {cfg.data.train.imgdir}"
    )

    train_files = [{keys[0]: img, keys[1]: seg}
                   for img, seg in zip(images[:n_train], labels[:n_train])]
    val_files = [{keys[0]: img, keys[1]: seg}
                 for img, seg in zip(images[-n_val:], labels[-n_val:])]

    # create a training data loader
    batch_size = cfg.batch_size
    logging.info(f"Batch size: {batch_size}")

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

    logging.info(f"Optimizer: {cfg.optimizer.name}")
    logging.info(f"LR Scheduler: {cfg.scheduler.name}")
    logging.info(f"Criterion: {cfg.loss.name}")

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
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.run()


def infer(cfg, model):
    """
    run inference, the output folder will be "./output"
    """
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    image_folder = os.path.abspath(data_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "*.nii.gz")))
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = factory.get_inferer()
    saver = monai.data.NiftiSaver(output_dir=prediction_folder, mode="nearest")
    with torch.no_grad():
        for infer_data in infer_loader:
            # logging.info(f"segmenting {infer_data["image_meta_dict"]["filename_or_obj"]}")
            preds = inferer(infer_data[keys[0]].to(device), net)
            n = 1.0
            for _ in range(4):
                # test time augmentations
                _img = RandGaussianNoised(
                    keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                pred = inferer(_img.to(device), net)
                preds = preds + pred
                n = n + 1.0
                for dims in [[2], [3]]:
                    flip_pred = inferer(torch.flip(
                        _img.to(device), dims=dims), net)
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0
            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()
            saver.save_batch(preds, infer_data["image_meta_dict"])

    # copy the saved segmentations into the required folder structure for submission
    submission_dir = os.path.join(prediction_folder, "to_submit")
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    files = glob.glob(os.path.join(prediction_folder, "volume*", "*.nii.gz"))
    for f in files:
        new_name = os.path.basename(f)
        new_name = new_name[len("volume-covid19-A-0"):]
        new_name = new_name[: -len("_ct_seg.nii.gz")] + ".nii.gz"
        to_name = os.path.join(submission_dir, new_name)
        shutil.copy(f, to_name)
    logging.info(f"predictions copied to {submission_dir}.")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    print('benchmark', torch.backends.cudnn.benchmark)

    torch.cuda.empty_cache()

    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')
