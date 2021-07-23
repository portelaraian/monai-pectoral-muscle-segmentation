from src.cnn.utils.logger import logger, log
from src.cnn.utils.config import Config
import argparse
import functools
from glob import glob
from multiprocessing import Pool
import os
import sys

import cv2
import nibabel as nb
import pandas as pd
from tqdm import tqdm

wdir = os.getcwd()
sys.path.insert(0, os.path.join(wdir, ".."))


def get_args():
    """Parse the arguments.

    Returns:
        parser: parser containing the parameters.
    """

    parser = argparse.ArgumentParser(
        description="Pectoralis Major Cross-Sectional Area Selection"
    )
    parser.add_argument("config")
    parser.add_argument("--n_pool", type=int, default=4)

    return parser.parse_args()


def main():
    args = get_args()
    cfg = Config.fromfile(args.config)
    cfg.n_pool = args.n_pool

    logger.setup(
        cfg.prediction_folder,
        name='postprocess_model_%s_config' % (cfg.model.name)
    )

    log(f"Model: {cfg.model.name}")

    masks = glob(
        os.path.join(cfg.data.test.imgdir, "masks/*.nii")
    )
    masks = [os.path.basename(mask) for mask in masks]

    with Pool(cfg.n_pool) as pool:
        records = list(tqdm(
            iterable=pool.imap_unordered(
                functools.partial(
                    pectoralis_major_muscle_selection,
                    test_workdir=os.path.join(cfg.data.test.imgdir, "masks"),
                    out_dir=cfg.prediction_folder
                ),
                masks
            ),
            total=len(masks),
        ))

    records = pd.DataFrame(records)
    print(records.tail)

    records.to_csv(
        f"{cfg.prediction_folder}/postprocessing_results.csv",
        index=False
    )


def pectoralis_major_muscle_selection(mask_id, test_workdir, out_dir):

    # Reading the masks (pred, label)
    pred_path = os.path.join(out_dir, mask_id)
    label_path = os.path.join(test_workdir, mask_id)

    pred = nb.load(pred_path).get_fdata()
    label = nb.load(label_path).get_fdata()

    record = {
        "id": mask_id,
        "y_true": 1
    }

    stats = {
        "slice_idx": [],
        "img_area": [],
        "pred_area_ratio": [],
        "label_area_ratio": [],
    }

    # Calculating the area ratio of the mask (pred and label)
    tota_slices = label.shape[-1]
    for _slice in range(tota_slices):
        slice_pred = pred[:, :, _slice, 1]
        slice_label = label[:, :, _slice]

        pixels_pred = cv2.countNonZero(slice_pred)
        pixels_label = cv2.countNonZero(slice_label)

        stats["slice_idx"].append(_slice)
        stats["img_area"].append(slice_pred.shape[0] * slice_pred.shape[1])

        stats["pred_area_ratio"].append(
            (pixels_pred / (slice_pred.shape[0] * slice_pred.shape[1])) * 100
        )
        stats["label_area_ratio"].append(
            (pixels_label /
             (slice_label.shape[0] * slice_label.shape[1])) * 100
        )

    # Saving some useful information
    stats = pd.DataFrame(stats)

    # 20150308.nii => 20150308
    mask_id = mask_id.split(".")[0]
    stats.to_csv(f"{out_dir}/{mask_id}.csv", index=False)

    # Comparing results (pred, label)

    # Remove this comment if you want to compare only the first major muscle.
    # max_label_slice = stats.iloc[
    #     stats["label_area_ratio"].idxmax()
    # ]["slice_idx"]

    # Top-3 major muscle (manually segmented, label)
    top3_max_label_slices = list(
        stats.sort_values(
            "label_area_ratio",
            ascending=False
        )["slice_idx"][:3]
    )

    # Top-3 major muscle (model segmentation, prediction)
    top3_max_pred_slices = list(
        stats.sort_values(
            "pred_area_ratio",
            ascending=False
        )["slice_idx"][:3]
    )

    record.update(
        {
            "y_score": [1 if max_label_slice in top3_max_pred_slices else 0 for max_label_slice in top3_max_label_slices]
        }
    )

    return record


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')
