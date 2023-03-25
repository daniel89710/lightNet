# LightNet

LightNet is a deep learning framework based on the popular darknet platform, designed to create efficient and high-speed Convolutional Neural Networks (CNNs) for computer vision tasks. The framework has been improved and optimized to provide a more versatile and powerful solution for various deep learning challenges.

## Table of Contents

-   [Key Features](#key-features)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Examples](#examples)
-   [License](#license)

## Key Features

LightNet incorporates several cutting-edge techniques and optimizations to improve the performance of CNN models. The main features include:

-   Semantic Segmentation Learning
-   2:4 Structured Sparsity
-   Channel Pruning
-   Post Training Quantization (Under Maintenance)
-   SQLite Log Storage

### Semantic Segmentation Learning

LightNet has been extended to support semantic segmentation learning, which allows for more accurate and detailed segmentation of objects within an image. This feature enables the training of CNN models to recognize and classify individual pixels in an image, allowing for more precise object detection and scene understanding.

For example, semantic segmentation can be used to identify individual objects within an image, such as cars or pedestrians, and label each pixel in the image with the corresponding object class. This can be useful for a variety of applications, including autonomous driving and medical image analysis.

### 2:4 Structured Sparsity

The 2:4 structured sparsity technique is a novel method for reducing the number of parameters in a CNN model while maintaining its performance. This approach enables the model to be more efficient and requires less computation, resulting in faster training and inference times.

For example, using 2:4 structured sparsity can reduce the memory footprint and computational requirements of a CNN model, making it easier to deploy on resource-constrained devices such as mobile phones or embedded systems.

### Channel Pruning

Channel pruning is an optimization technique that reduces the number of channels in a CNN model without significantly affecting its accuracy. This method helps to decrease the model size and computational requirements, leading to faster training and inference times while maintaining performance.

For example, channel pruning can be used to reduce the number of channels in a CNN model for image classification, while still maintaining a high level of accuracy. This can be useful for deploying models on devices with limited computational resources.

### Post Training Quantization (Under Maintenance)

Post training quantization is a technique for reducing the memory footprint and computational requirements of a trained CNN model. This feature is currently under maintenance and will be available in a future release.

### SQLite Log Storage

LightNet supports storing training logs in SQLite databases, making it easier to analyze and visualize training progress over time. This feature enables users to efficiently manage their training logs and better understand the performance of their models.

## Installation

Please follow the darknet installation instructions to set up LightNet on your machine. Additionaly, you need install sqlite3-dev.

```
sudo apt-get install libsqlite3-dev
```

## Usage

You can use LightNet just like you would use darknet. The command line interface remains the same, with additional options and features for the new improvements. For a comprehensive guide on using darknet, please refer to the official darknet documentation.
As for advanced usage, let's wait until the next release. Stay tuned!


## Examples

You can find examples of using LightNet's features in the examples directory. These examples demonstrate how to use the new features and optimizations in LightNet to train and test powerful CNN models.

### Inference for Detection
```
./lightNet detector [test/demo] data/bdd100k.data cfg/lightNet-BDD100K-1280x960.cfg weights/lightNet-BDD100K-1280x960.weights [image_name/video_name]
```

### Inference for Segmentation
```
/lightNet segmenter [test/demo] data/bdd100k.data cfg/lightSeg-BDD100K-laneMarker-1280x960.cfg weights/lightSeg-BDD100K-laneMarker-1280x960.weights [image_name/video_name]
```
## Results

### Results on BDD100K

| Model | Resolutions | GFLOPS | Params | mAP50 | AP@car| AP@person | cfg | weights |
|---|---|---|---|---|---|---|---|---|
| lightNet | 1280x960 | 58.01 | 9.0M | 55.7 | 81.6 | 67.0| [github](https://github.com/daniel89710/lightNet/blob/master/cfg/lightNet-BDD100K-1280x960.cfg) |[GoogleDrive](https://drive.google.com/file/d/1qTBQ0BkIYqcyu1BwC54_Z9T1_b702HKf/view?usp=sharing) |
| yolov8x | 640x640 | 246.55 | 70.14M | 55.2 | 80.0 | 63.2 | [github](https://github.com/daniel89710/lightNet/blob/master/cfg/yolov8x-BDD100K-640x640.cfg) | [GoogleDrive](https://drive.google.com/file/d/1hrHeugq0-mL6EtxUAi-rkfrzg6KwgCQQ/view?usp=sharing)|
 
 

## License

LightNet is released under the same YOLO license as darknet. You are free to use, modify, and distribute the code as long as you retain the license notice.
