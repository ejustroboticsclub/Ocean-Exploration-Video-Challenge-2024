# Ocean Exploration Video Challenge 2024

This repository showcases the development of a deep learning model for detecting brittle stars in underwater images using the YOLOv8 (You Only Look Once) object detection framework. This project was created for the Ocean Exploration Video Challenge, organized by the MATE ROV Competition in partnership with NOAA Ocean Exploration. The challenge aims to streamline the annotation process of organisms in remotely operated vehicle (ROV) dive videos, focusing specifically on the continuous tracking and annotation of brittle stars. Accurate detection and tracking of these marine invertebrates are crucial for monitoring biodiversity and assessing the health of marine ecosystems. By participating in this competition, the project contributes to advancing AI-driven tools that aid in the exploration and understanding of our oceans.

## Table of Contents
1. [Dataset](#dataset)
1. [Scripts](#scripts)
1. [Requirements](#requirements)
1. [Installation](#installation)
1. [Usage](#usage)
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

## Scripts

- **`predict.py`**: </br> Contains the `Predictor` class, which handles the detection and bounding box drawing functionalities for both images and videos.
- **`app.py`**: </br> Contains the `App` class, which builds a Tkinter-based GUI for uploading and processing images and videos.

___
## Requirements

- Python 3.9
- ultralytics
- opencv-python
- numpy
- pillow
- tkinter (included with Python standard library)

___

## Installation
1. Create a new environment with a 3.9 Python version.
1. Clone the repository.
   ```
   git clone https://github.com/ejustroboticsclub/Ocean-Exploration-Video-Challenge-2024.git
   ```
1. Navigate to the `Ocean-Exploration-Video-Challenge-2024` directory.
   ```
   cd Ocean-Exploration-Video-Challenge-2024
   ```
1. Download the [weights file](https://www.kaggle.com/models/amromeshref/brittle-stars-detection/) (`last.pt`) of the trained model and put it in the `weights` directory.
1. Type the following command to install the requirements file using pip:
   ```
   pip install -r requirements.txt
   ```
1. Type the following command to use the GUI app:
   ```
   python3 app.py
   ```
___

## Usage

#### Processing an Image
You can use the command-line interface to process images directly:
```
python3 predict.py --type image --path path/to/image.jpg --conf-threshold 0.1 --show-conf True
```
- `conf-threshold`: Sets the confidence threshold for detection. Only detections with a confidence score higher than this value will be considered. You can adjust this value depending on how strict you want the detection to be.
- `show-conf`: If set to True, the confidence scores will be displayed on the bounding boxes drawn around detected brittle stars.
- The results will be saved at `runs/images`. The `runs` directory will be created automatically.


</br>

#### Processing a Video
You can use the command-line interface to process videos directly:
```
python3 predict.py --type video --path path/to/video.mp4 --conf-threshold 0.1 --show-conf True
```
- The results will be saved at `runs/videos`. The `runs` directory will be created automatically.


</br>

#### Using the GUI
To launch the Tkinter GUI:
```
python3 app.py
```
In the GUI, you can:
- Upload Image: Upload an image and process it for brittle star detection.
- Upload Video: Upload a video and process it for brittle star detection.
- View Result: View the processed image or video. The processed files are saved in the `runs/images` and `runs/videos` directories.


#### Using Code Snippet

To run the model on an input image and draw bounding boxes on it:
```python
from predict import Predictor
import cv2

# Create a Predictor object
predictor = Predictor()

# Define the path to the image
image_path = "/path/to/image.jpg"

# Read the image using OpenCV
image = cv2.imread(image_path)

# Run the model on the image
results = predictor.predict(image)

# Draw bounding boxes on the image
image_with_boxes = predictor.draw_bounding_boxes(image, results, conf_threshold=0.1, show_conf=True)

# Display the image with bounding boxes
cv2.imshow("image", image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

</br>
To run the model on an input video and draw bounding boxes on it:
```python
from predict import Predictor
import cv2

# Create a Predictor object
predictor = Predictor()

# Define the path to the input video
video_path = "path/to/video.mp4"

# Read the input video and process it frame by frame
video = cv2.VideoCapture(video_path)
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Predict and draw bounding boxes
    results = predictor.predict(frame)
    frame_with_boxes = predictor.draw_bounding_boxes(frame, results, conf_threshold=0.1, show_conf=True)

    # Display the frame with bounding boxes
    cv2.imshow('frame', frame_with_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects
video.release()
cv2.destroyAllWindows()
```
