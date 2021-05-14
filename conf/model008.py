model_id = "UNet_focalLoss_192dim_batch2"
workdir = './model/model008'
seed = 950


epochs = 1000
amp = True
batch_size = 2
num_workers = 4
imgsize = (192, 192, 16)

# Inferer
prediction_folder = f"{workdir}/output"
checkpoints = f"{workdir}/*.pt"

loss = dict(
    name='DiceFocalLoss',
    params=dict(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
    ),
)

optimizer = dict(
    name='Adam',
    params=dict(
        lr=0.0005,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.00002
    ),
)

model = dict(
    name='UNet',
    params=dict(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ),
)

scheduler = dict(
    name='CosineAnnealingScheduler',
    params=dict(
        param_name='lr',
        start_value=1e-5,
        end_value=1e-3,
    ),
)

data = dict(
    train=dict(
        imgdir='./input/train/version_3',
        imgsize=imgsize,
        batch_size=batch_size,
        loader=dict(
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
    ),

    valid=dict(
        imgdir='./input/train/version_3',
        imgsize=imgsize,
        batch_size=1,
        loader=dict(
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
    ),

    test=dict(
        imgdir='./input/test',
        imgsize=imgsize,
        batch_size=1,
        loader=dict(
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    )
)
