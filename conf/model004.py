workdir = './model/SegResNet_v5'
seed = 9233


epochs = 1000
amp = True
batch_size = 8
num_workers = 4
imgsize = (192, 192, 16)

train_frac = 0.80
val_frac = 0.2

prediction_folder = f"{workdir}/output"
checkpoints = f"{workdir}/*.pt"

loss = dict(
    name='DiceCELoss',
    params=dict(),
)

optimizer = dict(
    name='Adam',
    params=dict(
        lr=0.0002,
        betas=(0.9, 0.999),
        eps=1e-08,
    ),
)

model = dict(
    name='SegResNet',
    params=dict(
        spatial_dims=3,
        init_filters=8,
        in_channels=1,
        out_channels=2,
        dropout_prob=0.2,
        norm_name='group',
        num_groups=8,
        use_conv_final=True,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
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
        imgdir='./input/train/',
        imgsize=imgsize,
        batch_size=batch_size,
        loader=dict(
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
    ),

    valid=dict(
        imgdir='./input/train/',
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
