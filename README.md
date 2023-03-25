lightNet

lightNet is an open-source deep learning framework based on AlexeyAB's darknet, with additional improvements to enhance its capabilities and performance. This repository aims to provide an easy-to-use and efficient platform for various computer vision tasks while paying homage to the original darknet and its contributors, including YOLO's creator Joseph Redmon and Alexey Bochkovskiy.
Improvements

The following improvements have been implemented in lightNet:

    Semantic Segmentation: The framework now supports training models for semantic segmentation tasks, allowing for pixel-wise classification of images.
	    2:4 Structured Sparsity: lightNet incorporates 2:4 structured sparsity, which enables the creation of more efficient and compact models by reducing redundancy in the neural network's structure.
		    Channel Pruning: To further optimize the network, channel pruning has been added, which removes less significant channels from the model, leading to a lighter and faster network without sacrificing accuracy.
			    Post Training Quantization (under maintenance): Post-training quantization is available for further compression and acceleration of trained models, reducing the memory footprint and computational resources required during inference.
				    SQLite Logging: lightNet supports logging training and validation metrics to SQLite databases, enabling better organization and analysis of the training process.
					
					Getting Started
					
					Please refer to the original darknet repository for detailed instructions on how to set up and use the framework. The same procedures apply to lightNet, with the additional features and improvements mentioned above.
					License
					
					This project is released under the YOLO license, which permits unrestricted use, distribution, and modification. If we meet someday and you think the work is worth it, feel free to buy us a beer in return. See the LICENSE file for more information.
