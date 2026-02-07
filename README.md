# YOLO-LXR: An Enhanced Model for Pathology Detection in Chest X-Rays
This repository contains the implementation of a YOLO-inspired deep learning architectures designed for detecting lung nodules in X-ray and CT scan images. The models have been developed to enhance automated diagnostic accuracy and assist radiologists in identifying lung abnormalities efficiently.

## Getting Started

Follow the steps below to set up and use this repository effectively:

### 1. Download and Upload

Clone or download the contents of this repository.

Upload the extracted files to your Google Drive for further processing.

### 2. Dataset Preparation

Download the required datasets from the cited sources.

Apply necessary preprocessing techniques to prepare the data for training.

Ensure that the dataset folders are correctly named as follows:

- Rename the VinDr-CXR dataset folder to: VinDr-CXR

- Rename the X-Nodule dataset folder to: X-Nodule

- Rename the ChestX-ray14 dataset folder to: ChestX-ray14

By following these steps, you will ensure that the models can access the datasets correctly and operate without compatibility issues.

## Documentation

After downloading the datasets, create the corresponding *.yaml* configuration files that define the dataset structure and guide the model architecture to the correct data paths.
Two examples are provided for reference:

- VinDr-CXR-data.yaml

- data.yaml

These files specify the locations of the training, validation, and testing data for each dataset.

The repository includes three implementation notebooks, each tailored for a specific dataset. These notebooks handle model training, validation, and evaluation. All are designed to run seamlessly in Google Colab.

- *CestX_ray14_Implementation.ipynb* – Implementation for the ChestX-ray14 dataset

- *VinDr_CXR_Implementation.ipynb* – Implementation for the VinDr-CXR dataset

- *X-Nodules_Implementation.ipynb* – Implementation for the X-Nodules dataset

Each notebook ensures a streamlined workflow, from data preparation to model evaluation, making it easy to reproduce and compare results across datasets.

## Additional Information
*Yolo for Lungs Results.xlsx* is an Excel file that brings together all the results from this research, organised into three sheets. While most of the information can also be found in the implementation files, this workbook provides everything in one place and in a more tidy and easy-to-read format.

Each sheet corresponds to a different dataset and includes a table showing the performance of each benchmark architecture. The first sheet also contains the ablation study for YOLOv5 and RepVGG-GELAN. The other two sheets include a comparison table and a summary table, which make it easier to compare model performance on multi-class datasets.

## License

This project is released under the MIT License.  
See the LICENSE file for details.

Please note that use of the code must comply with the licensing terms of any datasets employed.







