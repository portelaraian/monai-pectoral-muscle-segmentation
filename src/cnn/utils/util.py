def split_data(images, labels):
    train_files = []
    val_files = []

    for i in range(5):
        train_files.append(
            [
                {"image": img, "label": seg}
                for img, seg in zip(
                    images[: (15 * i)] + images[(15 * (i + 1)):],
                    labels[: (15 * i)] + labels[(15 * (i + 1)):],
                )
            ]
        )
        val_files.append(
            [
                {"image": img, "label": seg}
                for img, seg in zip(images[(15 * i): (15 * (i + 1))],
                                    labels[(15 * i): (15 * (i + 1))])
            ]
        )

    return train_files, val_files
