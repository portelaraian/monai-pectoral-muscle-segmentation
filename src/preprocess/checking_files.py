import glob
import argparse
import shutil

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-mri', type=str)
    parser.add_argument('--input-masks', type=str)
    return parser.parse_args()

def remove_suffix(masks):
    """Remove maskAXIAL suffix from masks' filenames
    
    Args:
        masks (list):  list of masks filenames (glob)
    
    Returns:
        None
    """
    for mask in masks:
        shutil.move(mask, mask.replace("maskAXIAL", ""))

def comparisons(mris, masks):
    """Compare filenames and check if there's any missing masks
    
    Args:
        mris (list):  list of MRIs (glob)
        masks (list): list of masks (glob)
    
    Returns:
        matches (list)
    """
    # Checking quantities (both)
    print(f"[Comparisons]: Total {len(mris)} MRIs")
    print(f"[Comparisons]: Total {len(masks)} masks\n")

    # One by one
    mris = [mri.split("/")[-1].split(".")[0] for mri in mris]
    masks = [mask.split("/")[-1].split(".")[0] for mask in masks]

    missing_masks = [mask for mask in masks if mask not in mris]
    missing_mri   = [mri for mri in mris if mri not in masks]
    matches       = [mri for mri in mris if mri in masks]

    print(f"[Comparisons]: Total missings {len(missing_mri)} MRIs")
    print(missing_mri)
    print("-"*10)
    print(f"\n[Comparisons]: Total missings {len(missing_masks)} masks")
    print(missing_masks)
    print("-"*10)
    print(f"\n[Comparisons]: Total matches {len(matches)} masks")
    print(matches)

    return matches

def segregate_matches(matches):
    """Move files to another folder (organization purposes)
    
    Args:
        mris (list): list of matches

    Returns:
        None
    """
    original_path = "./input/raw"
    mask_path     = "./input/masks"
    MRIs_path     = "./input/mri"

    index = 1
    for match in matches:
        try:
            shutil.move(f"{original_path}/mri/{match}.nii.gz", f"{MRIs_path}/{index}_{match}.nii.gz")
            shutil.move(f"{original_path}/masks/{match}.nii", f"{mask_path}/{index}_{match}.nii")

            index += 1
        except Exception as e:
            print(e)
            return False
    return True

def main(args):
    """Main

    Args:
        args (argparse)
    Returns:
        None
    """
    
    files = glob.glob(f"{args.input_mri}*.gz")
    masks = glob.glob(f"{args.input_masks}*.nii")
    
    remove_suffix(masks)
    matches = comparisons(files, masks)
    if len(matches) > 0:
        if segregate_matches(matches):
            print("[Segregate files]: Moved files successfully...")
        else:
            print("[Segregate files]: An error occured...")
 
if __name__ == "__main__":
    args = get_args()
    
    main(args)