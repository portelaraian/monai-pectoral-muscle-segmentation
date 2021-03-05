workdir = './model/SegResNet'
seed = 333

epochs = 500
amp = True
batch_size = 8
num_workers = 4
imgsize = (192, 192, 16)

loss = dict(
    name='BCEWithLogitsLoss',
    params=dict(),
)

optimizer = dict(
    name='Adam',
    params=dict(
        lr=0.0005,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.00002,
    ),
)

model = dict(
    name='SegResNet',
    params=dict(
        spatial_dims=3,
        init_filters=8,
        in_channels=1,
        out_channels=n_classes,
        dropout_prob=None,
        norm_name='group',
        num_groups=8,
        use_conv_final=True,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    ),
)

scheduler = dict(
    name='MultiStepLR',
    params=dict(
        milestones=[1, 2],
        gamma=3/7,
    ),
)

data = dict(
    train=dict(
        imgdir='./input/train',
        imgsize=imgsize,
        batch_size=batch_size
        loader=dict(
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        ),
    ),
    test=dict(
        imgdir='./input/test',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    )
)
