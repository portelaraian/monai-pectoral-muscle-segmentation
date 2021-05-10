from monai.data import partition_dataset
import itertools


class SplitDataset():
    """Split the dataset into multiple folds
    """

    def __init__(self, files, seed, n_folds=5):
        self.data = partition_dataset(
            data=files,
            num_partitions=n_folds,
            shuffle=True,
            seed=seed,
        )

    def get_data(self, current_fold, keys, path_to_masks_dir):
        """Get the data and run kfold.

        Args:
            data (data splitted): list returned from split_data function.
            current_fold (int): interger indicating the current fold.

        Returns:
            list: list containing dictionary paired data separated into folds. 
        """
        folds = [fold for fold in range(5) if fold != current_fold]

        train_files_image = list(
            itertools.chain(
                self.data[folds[0]],
                self.data[folds[1]],
                self.data[folds[2]],
                self.data[folds[3]]
            )
        )
        val_files_image = self.data[current_fold]

        # using the same id from train_files_image and val_files_image to set the masks path
        # train_file.split('/')[-1].split('.')[0] is the image/mri id. (e.g. 20170729135310_5.nii.gz)
        train_files_label = [
            f"{path_to_masks_dir}/{file.split('/')[-1].split('.')[0]}.nii" for file in train_files_image
        ]
        val_files_label = [
            f"{path_to_masks_dir}/{file.split('/')[-1].split('.')[0]}.nii" for file in val_files_image
        ]

        train_files = [{keys[0]: img, keys[1]: seg}
                       for img, seg in zip(train_files_image, train_files_label)]
        val_files = [{keys[0]: img, keys[1]: seg}
                     for img, seg in zip(val_files_image, val_files_label)]

        return train_files, val_files
