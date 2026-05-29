### Author
Shirui, Guo

### Description
Official PyTorch implementation of OCM: A deep-learning Python module for reconstructing 3D volumes from encoded 2D measurements. Code for the paper "Off-axis Compressed Encoding for High-Resolution, High-Throughput, High-Contrast Volumetric Microscopy".

### Software

* Python 3.11 (tested on)
* Conda
* PyTorch 2.3.0 (tested on)
* Windows 10 / 11
* PyCharm 2022.3+

### Hardware

* CPU or GPU that supports CUDA, CuDNN, and PyTorch 2.3.0.
* Minimum requirement: 12 GB VRAM (e.g., TITAN). Recommended: 24 GB VRAM (e.g., NVIDIA GeForce RTX 3090 ).

## Instructions

* Install PyTorch and other dependencies. The main Python dependencies include:

```text
numpy
pytorch
tifffile
matplotlib
torchsummary
```

### Usage

1. For a given image stack (tiff format), first run the python code `Preprocess_main.py` to process the raw datasets,  generate the `.npy` format training data.
   * **Input parameters:** `file_root`, `save_file`, `sample_number`, `channels`, `arrays`, `point_1` (crop coordinates), `layers`, `speed`

2. Verify or adjust the normalization coefficients and dataset loading parameters in `data_loader_our.py`.
   * **Input parameters:** `layer`, `strip`, `speed`, normalization scaling factors (`self.input` / `self.output` divisors)

3. Run `train_our.py` to train the deep learning reconstruction network (e.g., LW-UNet, ResUNet, or RCAN).
   * **Input parameters:** `--train_save_file`, `--data_file`, `--data_type`, `--train_test_val`, `--channel_in`, `--channel_out`

4. Run `predict_our.py` (or `predict_our_oridata.py`) for the reconstruction of the testing image stack. This utilizes the trained network to restore structural details across the axial depth.
   * **Input parameters:** `file` (test dataset path), `file_write` (output restoration path), `model` (weights load path), `scale` (normalization coefficient), `patch_size`, `stride`, `speed`, `depth`
