# Installation

## Requirements

Currently, NVIDIA GPUs are required to use this library.

- NVIDIA driver, CUDA toolkit is required. Please refer to [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for installation.
- OpenCV is required for processing images.
- SQLite3 is required for training logs.

```bash
sudo apt install libopencv-dev libsqlite3-dev
```

## Build

```bash
git clone https://github.com/yunkai1841/lightnet.git
cd lightnet
make -j
```