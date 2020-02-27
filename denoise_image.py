#!/usr/bin/env python
"""
denoise_image.py

1) Loads the saved model

2) Optional: checks and plots the model in the test directory

3) TODO Runs on the specified tif file.
"""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from csbdeep.utils import plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE
from tifffile import imread

if __name__ == "__main__":
    # Configure the command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base_dir",
        help="the base directory the input and output files are located in",
        required=True)
    parser.add_argument(
        "--model_dir",
        help="the directory to store the models in",
        default="models")
    parser.add_argument(
        "--model_name",
        help="the name of your model (will be stored in model_dir)",
        default="my_model")
    parser.add_argument(
        "--path_to_test_image",
        help="the path to the test high/low pair, if available",
        default="test")
    # parser.add_argument(
    #   "-o",
    #   "--output_filename",
    #   help="the name of the out"
    # )
    parser.add_argument("-p", "--plot", action="store_true", default=True)
    args = parser.parse_args()

    # Load the trained model
    model = CARE(config=None, name=args.model_name, basedir=args.model_dir)

    # Read the test images
    x = imread(
        os.path.join(args.base_dir, args.path_to_test_image, "low_snr.tif"))
    y = imread(
        os.path.join(args.base_dir, args.path_to_test_image, "high_snr.tif"))

    restored = model.predict(x, axes="YX")  # , n_tiles=(1,4,4))

    # Save the restored image
    save_tiff_imagej_compatible(
        os.path.join(args.base_dir, args.path_to_test_image, "predicted.tif"),
        restored,
        axes="YX")

    # Plot the restored image next to the test pair
    if args.plot:
        plt.figure(figsize=(16, 10))
        plot_some(
            np.stack([x, restored, y]),
            title_list=[['low res', 'CARE', 'target']])
        plt.show()
