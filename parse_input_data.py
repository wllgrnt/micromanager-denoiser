#!/usr/bin/env python
"""
parse_input_data.py

Takes the tif files generated from micro-manager and converts them to CSBDeep's favoured .npz
format.

Steps:
1) Copy the micro-manager output to the specified base_directory, into folders marked
"high_snr" and "low_snr" (params: high_snr_channel and low_snr_channel, base_dir, input_location)
TODO add the option to have multiple low_snr channels.

2) Moves one of those high_snr/low_snr pairs into a "test" folder for final validation

3) Uses the RawData object and create_patches to generate the model training data as a .npz file.
TODO some data augmentation via rotations, shears etc
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
import argparse

from tifffile import imread
from csbdeep.utils import plot_some
from csbdeep.data import RawData, create_patches


def generate_output_dirs(base_dir: str):
    """
    Adds a low_snr and high_snr directory to the specified base_dir,
    if they don't already exist
    """
    low_snr_dir = os.path.join(base_dir, "low_snr")
    if not os.path.exists(low_snr_dir):
        os.mkdir(low_snr_dir)

    high_snr_dir = os.path.join(base_dir, "high_snr")
    if not os.path.exists(high_snr_dir):
        os.mkdir(high_snr_dir)


def move_images(input_directory: str,
                base_directory: str,
                channel_high: int,
                channel_low: int) -> int:
    """
    Moves the files from ${input_directory} into the high_snr and low_snr folders
    in matched pairs. The format of the MicroManager output is
    input_dir/1-Pos000_000/img_channel000_*.tif and
    input_dir/1-Pos000_000/img_channel001_*.tif

    The low_snr and high_snr pairs will have the same file name, but be located
    in different folders
    """
    for image_number, sub_dir in enumerate(
            glob.glob(os.path.join(input_directory, "*Pos*"))):
        # By default, channel 0 is low res, channel 1 is high res
        high_res_file_original_path = glob.glob(
            os.path.join(sub_dir, f"*channel{channel_high:03}*.tif"))
        # We only expect one match here
        assert len(high_res_file_original_path) == 1
        high_res_file_original_path = high_res_file_original_path[0]
        low_res_file_original_path = glob.glob(
            os.path.join(sub_dir, f"*channel{channel_low:03}*.tif"))
        assert len(low_res_file_original_path) == 1
        low_res_file_original_path = low_res_file_original_path[0]

        # Copy the files into the high_snr and low_snr directories.

        shutil.copy(high_res_file_original_path,
                    os.path.join(base_directory, "high_snr",
                                 f"img_{image_number}.tif"))
        shutil.copy(low_res_file_original_path,
                    os.path.join(base_directory, "low_snr",
                                 f"img_{image_number}.tif"))
    return image_number


if __name__ == "__main__":
    # Configure the command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        help="the location of the micro-manager tif files",
        required=True)
    parser.add_argument(
        "-b",
        "--base_dir",
        help="the base directory the images will be copied to",
        required=True)
    parser.add_argument(
        "--channel_high",
        help="the channel corresponding to the high-snr images",
        type=int,
        default=1)
    parser.add_argument(
        "--channel_low",
        help="the channel corresponding to the low-snr images",
        type=int,
        default=0)
    parser.add_argument(
        "-o",
        "--output_filename",
        help="the name of the npz file to write (defaults to data.npz)",
        default="data.npz")
    parser.add_argument("-p", "--plot", action="store_true", default=True)
    args = parser.parse_args()

    # Create the low_snr and high_snr directories
    generate_output_dirs(args.base_dir)
    print("output directories generated")
    # Move the images into the low_snr and high_snr directories
    number_of_images = move_images(
        input_directory=args.input_dir,
        base_directory=args.base_dir,
        channel_high=args.channel_high,
        channel_low=args.channel_low)
    print(f"{number_of_images} image pairs found")

    # Take a random one of the image pairs, hide as a final test image,
    # then optionally plot
    img_number = np.random.randint(0, number_of_images + 1)

    if args.plot:
        x = imread(
            os.path.join(args.base_dir, "low_snr", f"img_{img_number}.tif"))
        y = imread(
            os.path.join(args.base_dir, "high_snr", f"img_{img_number}.tif"))
        print(f"image size: {x.shape}")
        plt.figure(figsize=(16, 10))
        plot_some(
            np.stack([x, y]),
            title_list=[['low snr', 'high snr']], )
        plt.show()

    if not os.path.exists(os.path.join(args.base_dir, "test")):
        os.mkdir(os.path.join(args.base_dir, "test"))
    shutil.move(
        os.path.join(args.base_dir, "low_snr", f"img_{img_number}.tif"),
        os.path.join(args.base_dir, "test", "low_snr.tif"))
    shutil.move(
        os.path.join(args.base_dir, "high_snr", f"img_{img_number}.tif"),
        os.path.join(args.base_dir, "test", "high_snr.tif"))

    # Read the pairs, passing in the axis semantics
    raw_data = RawData.from_folder(
        basepath=args.base_dir,
        source_dirs=['low_snr'],
        target_dir='high_snr',
        axes='YX')

    # From the stacks, generate 2D patches, and save to the output_filename
    create_patches(
        raw_data=raw_data,
        patch_size=(128, 128),
        n_patches_per_image=512,
        save_file=os.path.join(args.base_dir, args.output_filename))
