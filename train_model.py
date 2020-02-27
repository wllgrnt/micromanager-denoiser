#!/usr/bin/env python
"""
train_model.py

1) Does a test-train split on the specified .npz file

2) Constructs a CARE model

3) Trains that model

4) Saves the model for later use.
"""

import argparse
import os
import matplotlib.pyplot as plt

from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict, plot_history, plot_some
from csbdeep.models import Config, CARE

if __name__ == "__main__":
    # Configure the command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base_dir",
        help="the base directory the images will be copied to",
        required=True)
    parser.add_argument(
        "-i",
        "--input_filename",
        help="the name of the npz file to read (defaults to data.npz)",
        default="data.npz")
    parser.add_argument(
        "-t",
        "--train_steps_per_epoch",
        help="the number of training steps to do in each epoch",
        type=int,
        default=500)
    parser.add_argument(
        "-n",
        "--num_epochs",
        help="the number of total epochs to train for",
        type=int,
        default=100)
    parser.add_argument(
        "--model_dir",
        help="the directory to store the models in",
        default="models")
    parser.add_argument(
        "--model_name",
        help="the name of your model (will be stored in model_dir)",
        default="my_model")

    parser.add_argument("-p", "--plot", action="store_true", default=True)
    args = parser.parse_args()

    # Read the npz file, split into training and validation sets
    (X, Y), (X_val, Y_val), axes = load_training_data(
        os.path.join(args.base_dir, args.input_filename),
        validation_split=0.1,
        verbose=True)

    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

    if args.plot:
        plt.figure(figsize=(12, 5))
        plot_some(X_val[:5], Y_val[:5])
        plt.suptitle(
            '5 example validation patches (top row: source, bottom row: target)'
        )
        plt.show()

    # Construct a CARE model, defining its configuration via a Config object
    config = Config(
        axes,
        n_channel_in,
        n_channel_out,
        probabilistic=False,  # We don't need detailed stats just yet
        train_steps_per_epoch=args.train_steps_per_epoch,
        train_epochs=args.num_epochs)
    print(config)
    vars(config)

    model = CARE(config, args.model_name, basedir=args.model_dir)

    # Use tensorboard to check the training progress with logdir = basedir
    history = model.train(X, Y, validation_data=(X_val, Y_val))

    if args.plot:
        plt.figure(figsize=(16, 5))
        plot_history(history, ['loss', 'val_loss'],
                     ['mse', 'val_mse', 'mae', 'val_mae'])
        plt.show()

        plt.figure(figsize=(12, 7))
        _P = model.keras_model.predict(X_val[:5])
        if config.probabilistic:
            _P = _P[..., :(_P.shape[-1] // 2)]
        plot_some(X_val[:5], Y_val[:5], _P, pmax=99.5)
        plt.suptitle('5 example validation patches\n'
                     'top row: input (source),  '
                     'middle row: target (ground truth),  '
                     'bottom row: predicted from source')
        plt.show()

    model.export_TF()
