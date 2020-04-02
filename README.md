# micromanager-denoiser

There are three projects:

1. Denoising 
2. Segmentation (finding a cell mask, and counting the beads within).
3. Automatic phenotyping

Here we do the denoising

## Denoising

This is an application of CSBDeep, a Content-Aware Restoration (CARE) model, to denoising the output from micro-manager.

### Relevant CSBDeep Links

- [The main GitHub repo](https://github.com/CSBDeep)
- [Python package docs](http://csbdeep.bioimagecomputing.com/doc/)
- [Example Jupyter notebooks](http://csbdeep.bioimagecomputing.com/examples/)
- [The Nature methods paper](https://www.nature.com/articles/s41592-018-0216-7)
- [Guide to Fiji installation](https://github.com/csbdeep/csbdeep_website/wiki/CSBDeep-in-Fiji)
- [The main website](http://csbdeep.bioimagecomputing.com/)

NB: The CSBDeep offer a Docker container you can run CSBDeep in. They also allow you to train your own models and export a trained TF graph
for use in Fiji through their Fiji plugin.


I've written three scripts, found here. They are:

- parse_input_data.py - takes the micro-manager output (which is a set of folders, one folder for each image, and within that folder, one channel for low-snr and one channel for high-snr) and converts it into a set of patches, saved to data.npz.
- train_model.py - trains the model on the data.npz, splitting into train and validation data. Outputs a TF model stored in models/my_model by default.
- denoise_image.py - loads the trained model, plots a test image to double-check that it's working properly, then denoises a target image and saves it wherever you'd like.

I've also written two Colab notebooks testing the denoiser on some cell pictures.
- [A Colab notebook copying the CSBDeep 3D denoising example](https://colab.research.google.com/drive/1I1B7OkujXm_-xKcdhk0483CjcyodiU9T)
- [A Colab notebook applying the CSBDeep to some of our cell images](https://colab.research.google.com/drive/1tDldtRB_V2Qy5nfW3cP87G6ciMzkrxcE)



### Setup (Windows)

Starting from a blank box with a clean Windows 10 install and a GPU.

- Install Anaconda
    - [www.anaconda.com/distribution/#download-section](https://www.anaconda.com/distribution/#download-section) (go 64-bit)
- Install tensorflow (version 1, not version 2) in a new conda environment, ensure the GPU is wired up correctly.
    - For this you'll need: NVIDIA GPU drivers, CUDA Toolkit (v10), cuDNN (v7) , Optional: TensorRT v5
    - CUDA Toolkit is [here](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork). Installing CUDA Toolkit will sort your drivers out. Check by running nvcc -V in a command prompt. 
    -  cuDNN is [here](https://developer.nvidia.com/cudnn). Requires a NVIDIA account. The scripts are tested on cuDNN v7.6.5. Guide here: [https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows)


Now create a new Conda environment:
```
conda create -n dl_env python=3.7
conda activate dl_env
pip install tensorflow-gpu==1.15
```


Finally, install csbdeep
```
pip install csbdeep
```

Then run the three scripts, using the provided Anaconda Prompt.

Todo:

- [ ]  Make parse_input_data.py more robust - currently the paths are hardcoded.
- [ ]  Allow multi-channel files.
- [ ]  Add the code to run on user-provided images - currently denoise_image.py just works on the pre-provided test image.
- [ ]  Refine the underlying Keras model - more training data, augmentation of the input data.
- [ ] Move to TF 2.0

