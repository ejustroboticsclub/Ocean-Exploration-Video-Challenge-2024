from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = "models/last.pt"

class Predictor:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def draw_bounding_boxes(self, image: np.ndarray, results) -> np.ndarray:
        """
        Draw bounding boxes and confidence scores on the original image.
        Args:
            image (np.ndarray): The original image.
            results (Results): The YOLOv8 results object containing bounding boxes and confidence scores.
        Returns:
            np.ndarray: The image with bounding boxes and confidence scores drawn.
        """
        image_copy = image.copy()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                bounding_boxes = box.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
                scores = box.conf.cpu().numpy()  # Confidence scores
        
                for bounding_box, score in zip(bounding_boxes, scores):
                    x1, y1, x2, y2 = map(int, bounding_box)
                    conf_text = f'{score:.2f}'

                    # Draw the bounding box
                    cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put the confidence score above the bounding box
                    cv2.putText(image_copy, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image_copy
    
    def predict(self, image: np.ndarray):
        """
        Detect objects in the input image.
        Args:
            image (np.ndarray): The input image.
        Returns:
            Results: The YOLO results object containing bounding boxes and confidence scores.
        """
        results = self.model(image)
        return results



# Load model
# model = YOLO("models/last.pt")




# cap = cv2.VideoCapture("videos/seafloor_footage.mp4")

# # Define codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
# fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width of the video frames
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of the video frames
# out = cv2.VideoWriter("output_with_boxes.mp4", fourcc, fps, (width, height))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform inference
#     results = model(frame)
#     frame = draw_bounding_boxes(frame, results)

#     # Write the frame with bounding boxes to the output video
#     out.write(frame)

#     # Display results (optional)
#     # cv2.imshow("frame", frame)
#     # if cv2.waitKey(1) & 0xFF == ord("q"):
#     #     break

# # Release everything if the job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()