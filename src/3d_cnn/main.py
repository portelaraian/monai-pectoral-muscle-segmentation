import glob
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_meandice
from monai.networks.layers import Norm
from monai.networks.nets import UNet, BasicUNet
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    RandGaussianNoised,
    RandFlipd,
    RandAffined,
    SpatialPadd,
)
from monai.utils import first, set_determinism



def main():
    train_images = sorted(glob.glob("./input/mri/*.nii.gz"))
    train_labels = sorted(glob.glob("./input/masks/*.nii"))

    print("Total train images: {}".format(len(train_images)))
    print("Total train masks: {}".format(len(train_labels)))

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]
    

    set_determinism(seed=0)
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="LPS"),
            Spacingd(keys=["image", "label"], pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],  a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True
            ),
            SpatialPadd(["image", "label"], spatial_size=(320, 320, -1), mode="reflect"),
            RandAffined(
                        ["image", "label"],
                        prob=0.15,
                        rotate_range=(-0.05, 0.05),
                        scale_range=(-0.1, 0.1),
                        mode=("bilinear", "nearest"),
                        as_tensor_output=False,
                    ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(320, 320, 16),
                num_samples=3,
            ),
            RandGaussianNoised(["image"], prob=0.15, std=0.01),
            RandFlipd(["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=2, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]   
    )
    
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
            ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=16)
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    
    
    device = torch.device("cuda:0")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    
    
    epoch_num = 600
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
    post_label = AsDiscrete(to_onehot=True, n_classes=2)

    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}", end="\r")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (320, 320, 16)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = post_pred(val_outputs)
                    val_labels = post_label(val_labels)
                    value = compute_meandice(
                        y_pred=val_outputs,
                        y=val_labels,
                        include_background=False,
                    )
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join("./model/unet_testing/", "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )

if __name__ == '__main__':
    print_config()
    
    main()
    
    pass