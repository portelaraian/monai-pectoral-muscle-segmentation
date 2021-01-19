import glob
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-mri', type=str)
    parser.add_argument('--input-masks', type=str)
    return parser.parse_args()

def main(args):
    """Compare filenames and check if there's any missing mask

    Args:
        args (argparse): Input and output directories of masks and MRIs
    """
    
    files = glob.glob(args.input_mri)
    masks = glob.glob(args.input_masks)
    
    
    
    pass

if __name__ == "__main__":
    args = get_args()
    
    main(args)