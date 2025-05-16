import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os
import sys
import io

# Set UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Load your pre-trained VGG model
model = tf.keras.models.load_model('C:/ML Dataset/Brain Tumor/final_vgg19_brain_tumor.keras', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class TumorPredictionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window title and dimensions
        self.title("Brain Tumor Prediction")
        self.geometry("500x600")
        self.config(bg="#2c2f33")  # Dark background

        # Styling variables
        self.font_style = ("Helvetica", 14, "bold")
        self.text_color = "#ffffff"  # White text
        self.button_color = "#7289da"  # Minimalistic button color
        self.button_text_color = "#ffffff"
        self.label_color = "#99aab5"  # Slightly lighter label color

        # Upload Image label
        self.image_label = Label(self, text="Please upload an MRI scan", font=self.font_style, fg=self.label_color, bg="#2c2f33")
        self.image_label.pack(pady=20)

        # Upload Button
        self.upload_button = tk.Button(self, text="Upload Image", command=self.upload_image, font=self.font_style,
                                       bg=self.button_color, fg=self.button_text_color, bd=0, activebackground="#677bc4", activeforeground=self.text_color)
        self.upload_button.pack(pady=10)

        # Prediction Result label
        self.result_label = tk.Label(self, text="", font=("Helvetica", 16, "bold"), fg=self.text_color, bg="#2c2f33")
        self.result_label.pack(pady=20)

        # Area to display the uploaded image
        self.image_display = tk.Label(self, bg="#2c2f33")
        self.image_display.pack(pady=20)

    def upload_image(self):
        # Open file dialog to choose an image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.jpeg;*.png")])
        if file_path:
            # Load and display the image
            img = Image.open(file_path)
            img = img.resize((224, 224))  # Resize image to match VGG model input size
            img_display = ImageTk.PhotoImage(img)

            # Update image label
            self.image_display.config(image=img_display)
            self.image_display.image = img_display

            # Convert image to array for model prediction
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Normalize image

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]

            # Ensure the result string is encoded correctly
            result_text = f"Predicted Tumor Type: {predicted_class}".encode('utf-8', 'replace').decode('utf-8')

            # Display the result
            self.result_label.config(text=result_text)

if __name__ == "__main__":
    app = TumorPredictionApp()
    app.mainloop()
