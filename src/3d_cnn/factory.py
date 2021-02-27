import numpy as np
import torch
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


def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform."""

    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=(
            "bilinear", "nearest")[: len(keys)]),
        ScaleIntensityd(keys, minv=0.0, maxv=1.0),
        # ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]

    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(192, 192, -1),
                            mode="reflect"),  # ensure at least 192x192
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
                                       spatial_size=(192, 192, 16),
                                       num_samples=3),
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
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


def get_net():
    """returns a unet model instance."""

    n_classes = 2
    net = monai.networks.nets.BasicUNet(
        dimensions=3,
        in_channels=1,
        out_channels=n_classes,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.2,
    )

    '''
    net = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=n_classes, # 2
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    '''
    return net


def get_loss(loss_name, loss_params):
    try:
        loss = getattr(monai.losses, loss_name)(**loss_params)
    except:
        print("Failed to import and load the loss function")


def get_dataloader(dataset, batch_sz, shuffle=False):
    return monai.data.DataLoader(
        dataset,
        batch_size=batch_sz,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )   


def get_inferer(_mode=None):
    """returns a sliding window inference instance."""

    patch_size = (192, 192, 16)
    sw_batch_size, overlap = 2, 0.5
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


