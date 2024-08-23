# Ocean Exploration Video Challenge 2024

This repository showcases the development of a deep learning model for detecting brittle stars in underwater images using the YOLOv8 (You Only Look Once) object detection framework. This project was created for the Ocean Exploration Video Challenge, organized by the MATE ROV Competition in partnership with NOAA Ocean Exploration. The challenge aims to streamline the annotation process of organisms in remotely operated vehicle (ROV) dive videos, focusing specifically on the continuous tracking and annotation of brittle stars. Accurate detection and tracking of these marine invertebrates are crucial for monitoring biodiversity and assessing the health of marine ecosystems. By participating in this competition, the project contributes to advancing AI-driven tools that aid in the exploration and understanding of our oceans.

## Table of Contents
1. [Dataset](#dataset)

___

## Dataset

The datasets used for this project were sourced from Roboflow, a platform providing various annotated datasets for computer vision tasks. The images in the dataset were captured underwater, containing various marine organisms, including brittle stars. These images were particularly valuable for training the YOLO models due to their realistic and challenging underwater conditions, such as varying lighting and complex backgrounds.
The specific datasets utilized are as follows:

- [Source 1](https://universe.roboflow.com/test-xsnip/mate-brittle-star-detection)
- [Source 2](https://universe.roboflow.com/raghad-abo-el-eneen/sea-creatures-detection)
- [Source 3](https://universe.roboflow.com/rowan-mohamed/optimized_result)
- [Source 4](https://universe.roboflow.com/noaa-wg5ah/brittle-stars)

### Dataset Versions

To explore different approaches for brittle star detection, two versions of the dataset were created, each with a distinct labeling strategy:
- **Version 1: Brittle Star Only** </br>
In this version, the dataset consists of images that contain brittle stars exclusively. The labeling focuses solely on the brittle stars, with bounding boxes drawn around them. This approach resulted in a single-class dataset, where the only class is "Brittle Star." This version is designed to train the model to specifically recognize and locate brittle stars in the images, ignoring other organisms present. The dataset was split as follows:
  - Total Images: 3,031
  - Training Set: 2,121 images (70%)
  - Validation Set: 606 images (20%)
  - Test Set: 304 images (10%)
  - This dataset is located at `data/external/version-01`


- **Version 2: Brittle Star and Not Brittle Star** </br>
The second version of the dataset introduces a two-class labeling scheme. In addition to labeling brittle stars, any other organism present in the images is labeled as "Not Brittle Star." This version of the dataset provides a more comprehensive approach, allowing the model to distinguish between brittle stars and other marine organisms. The "Not Brittle Star" class encompasses all non-brittle star entities, ensuring that the model can differentiate brittle stars from other similar-looking objects. The dataset was split as follows:
  - Total Images: 6,085
  - Training Set: 3,656 images (60%)
  - Validation Set: 1,218 images (20%)
  - Test Set: 1,211 images (20%)
  - This dataset is located at `data/external/version-02`

___
