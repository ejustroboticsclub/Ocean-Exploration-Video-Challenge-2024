import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import threading
from predict import Predictor
from datetime import datetime
import os
import platform


class App:
    def __init__(self, root, predictor: Predictor):
        self.root = root
        self.predictor = predictor
        self.root.title("Brittle Star Detection")
        self.root.configure(bg="black")

        # Create buttons for image and video upload with black background and red font color
        button_style = {"bg": "black", "fg": "red", "bd": 2,
                        "font": ("Helvetica", 16), "width": 20, "height": 2}

        self.upload_image_btn = Button(
            root, text="Upload Image", command=self.upload_image, **button_style)
        self.upload_image_btn.pack(pady=10)

        self.upload_video_btn = Button(
            root, text="Upload Video", command=self.upload_video, **button_style)
        self.upload_video_btn.pack(pady=10)

        # Button to view the result (initially disabled)
        self.view_result_btn = Button(
            root, text="View Result", command=self.view_result, state=tk.DISABLED, **button_style)
        self.view_result_btn.pack(pady=10)

        # Label to display status
        self.status_label = Label(root, text="", bg="black", fg="white")
        self.status_label.pack(pady=20)

        # Canvas to display images
        self.canvas = tk.Canvas(root, width=640, height=640, bg="black")
        self.canvas.pack()

        # Track the type of the last processed result
        self.last_operation = None

    def reset_gui(self):
        # Clear the canvas
        self.canvas.delete("all")

        # Reset status label
        self.status_label.config(text="")

        # Disable the View Result button
        self.view_result_btn.config(state=tk.DISABLED)

        # Reset last operation type
        self.last_operation = None

    def upload_image(self):
        # Reset GUI before starting a new operation
        self.reset_gui()

        file_path = filedialog.askopenfilename(
            filetypes=[("All files", "*.*")])

        if file_path:
            self.last_operation = "image"
            self.status_label.config(text="Loading...")
            threading.Thread(target=self.process_image,
                             args=(file_path,)).start()

    def upload_video(self):
        # Reset GUI before starting a new operation
        self.reset_gui()

        file_path = filedialog.askopenfilename(
            filetypes=[("All files", "*.*")])

        if file_path:
            self.last_operation = "video"
            self.status_label.config(text="Loading...")
            threading.Thread(target=self.process_video,
                             args=(file_path,)).start()

    def process_image(self, file_path):
        try:
            # Read the image
            image = cv2.imread(file_path)

            # Predict and draw bounding boxes
            results = self.predictor.predict(image)
            image_with_boxes = self.predictor.draw_bounding_boxes(
                image, results)

            # Convert the image to a format Tkinter can display
            image_pil = Image.fromarray(image_with_boxes)

            # Save the processed image at runs/images
            current_time = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
            image_name, _ = os.path.splitext(os.path.basename(file_path))
            os.makedirs("runs/images", exist_ok=True)
            self.result_image_path = "runs/images/" + \
                f"{image_name}_{current_time}" + ".jpg"
            image_pil.save(self.result_image_path)

            # Update the status label and enable the View Result button
            self.status_label.config(
                text="Processing complete. Click 'View Result' to see the image.")
            self.view_result_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.status_label.config(
                text="An error occurred in processing the image. Please try again.")

    def process_video(self, file_path):
        try:
            cap = cv2.VideoCapture(file_path)

            # Define the output video path
            current_time = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
            video_name, _ = os.path.splitext(os.path.basename(file_path))
            os.makedirs("runs/videos", exist_ok=True)
            output_path = "runs/videos/" + \
                f"{video_name}_{current_time}" + ".mp4"

            # Define codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create the VideoWriter object
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Predict and draw bounding boxes
                results = self.predictor.predict(frame)
                frame_with_boxes = self.predictor.draw_bounding_boxes(
                    frame, results)

                # Write the processed frame to the output video
                out.write(frame_with_boxes)

            cap.release()
            out.release()

            # Update the status label and enable the View Result button
            self.result_video_path = output_path
            self.status_label.config(
                text="Processing complete. Click 'View Result' to see the video.")
            self.view_result_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.status_label.config(
                text="An error occurred in processing the video. Please try again.")

    def view_result(self):
        if self.last_operation == "image":
            # Load and display the processed image on the canvas
            img = Image.open(self.result_image_path)
            # Resize the image to fit the canvas
            img = img.resize((640, 640), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk  # keep a reference to avoid garbage collection

        elif self.last_operation == "video":
            # Play the video using the system's default video player
            if platform.system() == "Windows":
                os.startfile(self.result_video_path)
            elif platform.system() == "Darwin":  # macOS
                os.system(f'open "{self.result_video_path}"')
            else:  # Linux and other platforms
                os.system(f'xdg-open "{self.result_video_path}"')

    def save_result_image(self):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            img = Image.open(self.result_image_path)
            img.save(save_path)

    def save_result_video(self):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if save_path:
            import shutil
            shutil.copy(self.result_video_path, save_path)


if __name__ == "__main__":
    predictor = Predictor()
    root = tk.Tk()
    app = App(root, predictor)
    root.mainloop()
