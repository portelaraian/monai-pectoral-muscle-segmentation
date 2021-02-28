import argparse
import glob
import logging
import os
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join('./')))

import factory
from utils.lr_schedulers import DiceCELoss

import monai
from monai.handlers import (
    CheckpointSaver,
    MeanDice,
    StatsHandler,
    ValidationHandler,
    LrScheduleHandler,
    HausdorffDistance,
    ROCAUC
)
from monai.metrics import DiceMetric, compute_roc_auc
from monai.transforms import AsDiscreted
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar

import torch.nn as nn
import torch





def get_args():
    parser = argparse.ArgumentParser(
        description="Runs the segmentation algorithm.")

    parser.add_argument(
        "mode", metavar="mode", default="train", choices=("train", "infer"), type=str, help="mode of workflow"
    )
    parser.add_argument("--data_folder", default="",
                        type=str, help="training data folder")
    parser.add_argument("--model_folder", default="./runs",
                        type=str, help="model folder")
    return parser.parse_args()


def train(data_folder="./input/train", model_folder="./runs"):
    """run a training pipeline."""

    images = sorted(glob.glob(os.path.join(data_folder, "mri/*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_folder, "masks/*.nii")))

    logging.info(
        f"training: image/label ({len(images)}) folder: {data_folder}")

    amp = True  # auto. mixed precision
    keys = ("image", "label")
    train_frac, val_frac = 0.8, 0.2
    n_train = int(train_frac * len(images)) + 1
    n_val = min(len(images) - n_train, int(val_frac * len(images)))

    logging.info(
        f"training: train {n_train} val {n_val}, folder: {data_folder}")

    train_files = [{keys[0]: img, keys[1]: seg}
                   for img, seg in zip(images[:n_train], labels[:n_train])]
    val_files = [{keys[0]: img, keys[1]: seg}
                 for img, seg in zip(images[-n_val:], labels[-n_val:])]

    # create a training data loader
    batch_size = 8
    logging.info(f"batch size {batch_size}")

    train_transforms = factory.get_xforms("train", keys)
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms)

    train_loader = factory.get_dataloader(train_ds, batch_size, True)

    # create a validation data loader
    val_transforms = factory.get_xforms("val", keys)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
    val_loader = factory.get_dataloader(val_ds, 1)

    # create BasicUNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = factory.get_net().to(device)
    max_epochs, lr = 400, 1e-5
    logging.info(f"epochs {max_epochs}, lr {lr}")
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # create evaluator (to be used to measure model quality during training)
    val_post_transform = monai.transforms.Compose(
        [AsDiscreted(keys=("pred", "label"), argmax=(
            True, False), to_onehot=True, n_classes=2)]
    )
    val_handlers = [
        ProgressBar(),
        CheckpointSaver(save_dir=model_folder, save_dict={"net": net}, save_key_metric=True, key_metric_n_saved=15),
    ]
    evaluator = monai.engines.SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=factory.get_inferer(),
        post_transform=val_post_transform,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=False, output_transform=lambda x: (x["pred"], x["label"])),
            # "val_HausdorffDistance": HausdorffDistance(include_background=False, output_transform=lambda x: (x["pred"], x["label"])),
            #"val_roc_auc": ROCAUC(to_onehot_y=True, softmax=True, output_transform=lambda x: (x["pred"], x["label"])),
        },
        val_handlers=val_handlers,
        amp=amp,
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingScheduler(opt, 'lr', 1e-5, 1e-3, len(train_loader))
    loss_params = dict(
        to_onehot_y=True,
        softmax=True, 
        jaccard=True,
    )
    loss = factory.get_loss("DiceLoss", loss_params)

    # evaluator as an event handler of the trainer
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        StatsHandler(tag_name="train_loss",
                     output_transform=lambda x: x["loss"]),
        # LrScheduleHandler(scheduler)
    ]

    trainer = monai.engines.SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=DiceCELoss(),
        inferer=factory.get_inferer(),
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp,
    )
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.run()


def infer(data_folder="./input/validation", model_folder="./runs", prediction_folder="output"):
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
            logging.info(
                f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
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
    torch.cuda.empty_cache()

    args = get_args()

    monai.config.print_config()
    monai.utils.set_determinism(seed=0)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if args.mode == "train":
        data_folder = args.data_folder or "./input/train"
        train(data_folder=data_folder, model_folder=args.model_folder)
    elif args.mode == "infer":
        data_folder = args.data_folder or "./input/validation"
        infer(data_folder=data_folder, model_folder=args.model_folder)
    else:
        raise ValueError("Unknown mode.")
