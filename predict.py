from ultralytics import YOLO
import cv2
import numpy as np
import argparse
from datetime import datetime
import os


MODEL_PATH = "weights/last.pt"
CONFIDENCE_THRESHOLD = 0.0


class Predictor:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def draw_bounding_boxes(self, image: np.ndarray, results, conf_threshold: float = CONFIDENCE_THRESHOLD, show_conf=True) -> np.ndarray:
        """
        Draw bounding boxes and confidence scores on the original image.
        Args:
            image (np.ndarray) - BGR format: The original image.
            results: The YOLOv8 results object containing bounding boxes and confidence scores.
            conf_threshold (float) - The confidence threshold to filter out weak detections.
            show_conf (bool) - Whether to show the confidence scores on the detected objects.
        Returns:
            np.ndarray - BGR format: The image with bounding boxes and confidence scores drawn.
        """
        image_copy = image.copy()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Filter out Not Brittle Stars detections
                if int(box.cls) != 0:
                    continue

                # Get the Bounding box coordinates (x1, y1, x2, y2)
                bounding_boxes = box.xyxy.cpu().numpy()

                # Get the confidence scores
                scores = box.conf.cpu().numpy()  

                for bounding_box, score in zip(bounding_boxes, scores):
                    # Filter out weak detections
                    if score < conf_threshold:
                        continue

                    x1, y1, x2, y2 = map(int, bounding_box)
                    conf_text = f'{score:.2f}'

                    # Draw the bounding box
                    cv2.rectangle(image_copy, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)

                    # Put the confidence score on the image if required
                    if show_conf:
                        cv2.putText(image_copy, conf_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image_copy

    def predict(self, image: np.ndarray):
        """
        Detect objects in the input image.
        Args:
            image (np.ndarray) - BGR format: The input image.
        Returns:
            Results: The YOLO results object containing bounding boxes and confidence scores.
        """
        results = self.model(image)
        return results

    def process_image(self, image_path: str, conf_threshold: float = CONFIDENCE_THRESHOLD, show_conf: bool = True) -> None:
        """
        Process the image at the specified path and save the image with bounding boxes.
        Args:
            image_path (str) - The path to the input image.
            conf_threshold (float) - The confidence threshold to filter out weak detections.
            show_conf (bool) - Whether to show the confidence scores on the detected objects.
        Returns:
            None
        """
        # Read the image and perform inference
        image = cv2.imread(image_path)
        results = self.predict(image)
        image_with_boxes = self.draw_bounding_boxes(
            image, results, conf_threshold, show_conf)

        # Save the image with bounding boxes
        current_time = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
        image_name, _ = os.path.splitext(os.path.basename(image_path))
        os.makedirs("runs/images", exist_ok=True)
        result_image_path = "runs/images/" + \
            f"{image_name}_{current_time}" + ".jpg"
        cv2.imwrite(result_image_path, image_with_boxes)
        print("Image saved at:", result_image_path)

        # Display the image with bounding boxes
        cv2.imshow("image", image_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self, video_path: str, conf_threshold: float = CONFIDENCE_THRESHOLD, show_conf: bool = True) -> None:
        """
        Process the video at the specified path and save the video with bounding boxes.
        Args:
            video_path (str) - The path to the input video.
            conf_threshold (float) - The confidence threshold to filter out weak detections.
            show_conf (bool) - Whether to show the confidence scores on the detected objects.
        Returns:
            None
        """
        cap = cv2.VideoCapture(video_path)

        # Define the codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the output video path
        current_time = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
        video_name, _ = os.path.splitext(os.path.basename(video_path))
        os.makedirs("runs/videos", exist_ok=True)
        output_path = "runs/videos/" + f"{video_name}_{current_time}" + ".mp4"

        # Create the VideoWriter object
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference
            results = self.predict(frame)
            frame = self.draw_bounding_boxes(
                frame, results, conf_threshold, show_conf)

            # Write the frame with bounding boxes to the output video
            out.write(frame)

            # Display results (optional)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        print("Video saved at:", output_path)

        # Release everything if the job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True,
                        choices=["image", "video"], help="Type of input: image or video.")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to the input image or video.")
    parser.add_argument("--conf-threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Confidence threshold to filter out weak detections.")
    parser.add_argument("--show-conf", type=bool, default=True,
                        help="Whether to show the confidence scores on the detected objects.")
    args = parser.parse_args()

    predictor = Predictor()
    if args.type == "image":
        predictor.process_image(args.path, args.conf_threshold, args.show_conf)
    else:
        predictor.process_video(args.path, args.conf_threshold, args.show_conf)
