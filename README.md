# Galaxy Diffusion

This repo uses the [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) package and trains the network with the [GalaxyMNIST](https://github.com/mwalmsley/galaxy_mnist) images.  The resulting network can be used to generate "fake" galaxy images the have the same distribution as the training set.

There are scripts for training with both the low-res and high-res GalaxyMNIST data sets, and scripts for training with or without the class labels.

The SLURM submission scripts are also included that I used with the SCIAMA HPC at the ICG.

For fun, there is also a `gz_font.py` file that creates fake galaxies that look like letters.

NOTE: The submissions scripts and the `gz_font.py` files have hard coded file paths, adjust as needed.
