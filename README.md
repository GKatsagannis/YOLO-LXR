# Lung Nodule Detection Using YOLO-like Architectures: A Study on X-ray and CT-Scan Imaging
This repository contains the implementation of two YOLO-inspired deep learning architectures designed for detecting lung nodules in X-ray and CT scan images. The models have been developed to enhance automated diagnostic accuracy and assist radiologists in identifying lung abnormalities efficiently.

## Getting Started

Follow the steps below to set up and use this repository effectively:

### 1. Download and Upload

Clone or download the contents of this repository.

Upload the extracted files to your Google Drive for further processing.

### 2. Dataset Preparation

Download the required datasets from the cited sources.

Apply necessary preprocessing techniques to prepare the data for training.

Ensure that the dataset folders are correctly named as follows:

- Rename the LUNA16 dataset folder to: Luna16.v1i.yolov8

- Rename the X-Nodule dataset folder to: X-Nodule

By following these steps, you will ensure that the models can access the datasets correctly and operate without compatibility issues.

## Documentation

The code should execute without issues, except for sections related to CAF-YOLO. To run those parts successfully, follow these steps:

1. Locate the relevant line of code by hovering over it, as shown in the screenshot below.

![CAF-YOLO Instructions 1](https://github.com/user-attachments/assets/ab60cada-79e9-408c-9a54-7ed313146ea0)


2. Press Control (Ctrl) + Left Click to open the referenced script.

![CAF-YOLO Instructions 2](https://github.com/user-attachments/assets/83142da0-94b9-472e-871b-85cebf7d4568)


3. Once the script opens, modify the highlighted section of the code to specify the correct path to the dataset folder you intend to use with CAF-YOLO.
(Example: _/content/drive/MyDrive/Luna16.v1i.yolov8_)

By ensuring the correct dataset path is set, you will enable seamless execution of CAF-YOLO within the framework.
## License

This project is intended for research and educational purposes. Please ensure compliance with dataset licensing terms when using this repository.

