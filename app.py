import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np
from predict import Predictor  # Importing the Predictor class

class App:
    def __init__(self, root, predictor):
        self.root = root
        self.predictor = predictor
        self.root.title("Brittle Star Detection")
        self.root.configure(bg="black")  # Set background to black

        # Create buttons for image and video upload with black background and red font color
        button_style = {"bg": "black", "fg": "red", "bd": 2}

        self.upload_image_btn = Button(root, text="Upload Image", command=self.upload_image, **button_style)
        self.upload_image_btn.pack(pady=10)

        self.upload_video_btn = Button(root, text="Upload Video", command=self.upload_video, **button_style)
        self.upload_video_btn.pack(pady=10)

        # Button to view the result (initially disabled)
        self.view_result_btn = Button(root, text="View Result", command=self.view_result, state=tk.DISABLED, **button_style)
        self.view_result_btn.pack(pady=10)

        # Label to display status
        self.status_label = Label(root, text="", bg="black", fg="white")
        self.status_label.pack(pady=20)

        # Canvas to display images
        self.canvas = tk.Canvas(root, width=600, height=400, bg="black")
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
        self.reset_gui()  # Reset GUI before starting a new operation
        file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if file_path:
            self.last_operation = "image"
            self.status_label.config(text="Loading...")
            threading.Thread(target=self.process_image, args=(file_path,)).start()

    def upload_video(self):
        self.reset_gui()  # Reset GUI before starting a new operation
        file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if file_path:
            self.last_operation = "video"
            self.status_label.config(text="Loading...")
            threading.Thread(target=self.process_video, args=(file_path,)).start()

    def process_image(self, file_path):
        # Read the image
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict and draw bounding boxes
        results = self.predictor.predict(image)
        image_with_boxes = self.predictor.draw_bounding_boxes(image, results)

        # Convert the image to a format Tkinter can display
        image_pil = Image.fromarray(image_with_boxes)
        self.result_image_path = "result_image.png"
        image_pil.save(self.result_image_path)

        self.status_label.config(text="Processing complete. Click 'View Result' to see the image.")
        self.view_result_btn.config(state=tk.NORMAL)

    def process_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        output_path = "result_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict and draw bounding boxes
            results = self.predictor.predict(frame)
            frame_with_boxes = self.predictor.draw_bounding_boxes(frame, results)

            # Write the processed frame to the output video
            out.write(frame_with_boxes)
            i+=1
            if i == 10:
                break

        cap.release()
        out.release()

        self.result_video_path = output_path
        self.status_label.config(text="Processing complete. Click 'View Result' to see the video.")
        self.view_result_btn.config(state=tk.NORMAL)

    def view_result(self):
        if self.last_operation == "image":
            # Load and display the processed image on the canvas
            img = Image.open(self.result_image_path)
            img = img.resize((600, 400), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk  # keep a reference to avoid garbage collection

        elif self.last_operation == "video":
            # Play the video using the system's default video player
            import os
            import platform
            if platform.system() == "Windows":
                os.startfile(self.result_video_path)
            elif platform.system() == "Darwin":  # macOS
                os.system(f'open "{self.result_video_path}"')
            else:  # Linux and other platforms
                os.system(f'xdg-open "{self.result_video_path}"')

    def save_result_image(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            img = Image.open(self.result_image_path)
            img.save(save_path)

    def save_result_video(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if save_path:
            import shutil
            shutil.copy(self.result_video_path, save_path)

# Example usage:
if __name__ == "__main__":
    predictor = Predictor()
    root = tk.Tk()
    app = App(root, predictor)
    root.mainloop()
