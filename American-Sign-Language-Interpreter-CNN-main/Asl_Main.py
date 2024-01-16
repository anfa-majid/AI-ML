import os
import cv2
import numpy as np
from tkinter import Tk, Toplevel, Button, Entry, Label, StringVar, filedialog, messagebox
from tkinter.ttk import Frame, Label, Button, Style
from ttkthemes import ThemedTk
from keras.models import load_model
from keras.preprocessing import image

# Load the saved model
model = load_model('asl_cnn_model.h5')

# Preprocess image
def preprocess_image(image_path=None, img=None):
    if image_path:
        img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=0)
    return img

# Classify image
def classify_image(img):
    predictions = model.predict(img)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][class_index]
    return class_index, confidence

def on_feedback_click(is_correct, class_index, img, feedback_window):
    if not is_correct:
        # Handle incorrect prediction
        show_correct_label_window(class_index, img)
    feedback_window.destroy()

def show_correct_label_window(class_index, img):
    correct_label_window = Toplevel(root)
    correct_label_window.title("Correct Label")

    Label(correct_label_window, text="Enter the correct label:", font=("Arial", 14, "bold")).pack(pady=10)

    label_var = StringVar()
    entry = Entry(correct_label_window, textvariable=label_var, font=("Arial", 14))
    entry.pack(pady=10)

    Button(correct_label_window, text="Submit", command=lambda: on_correct_label_submit(class_index, img, label_var, correct_label_window)).pack(pady=10)

def on_correct_label_submit(class_index, img, label_var, correct_label_window):
    correct_label = label_var.get().strip().upper()

    if correct_label:
        # Map the correct label to its corresponding class index
        class_labels = ['A', 'B', 'C']  # Update this list with all the ASL alphabet labels that your model can predict
        correct_class_index = class_labels.index(correct_label)

        # One-hot encode the correct class index
        correct_class_encoded = np.zeros((1, len(class_labels)))
        correct_class_encoded[0, correct_class_index] = 1

        # Fine-tune the model using the image and the correct label
        model.fit(img, correct_class_encoded, epochs=1, verbose=0, batch_size=1)

    correct_label_window.destroy()

# Function to handle button click
def on_button_click():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img = preprocess_image(file_path)
    class_index, confidence = classify_image(img)

    # Replace this line with the list of your ASL alphabet labels
    class_labels = ['A', 'B', 'C']  # You should update this list with all the ASL alphabet labels that your model can predict

    messagebox.showinfo("Result", f"Predicted Class: {class_labels[class_index]}\nConfidence: {confidence * 100:.2f}%")

    feedback_window = Toplevel(root)
    feedback_window.title("Feedback")
    Label(feedback_window, text="Was the prediction correct?", font=("Arial", 14)).pack(pady=10)

    Button(feedback_window, text="Yes", command=lambda: on_feedback_click(True, class_index, img, feedback_window)).pack(side='left', padx=10, pady=10)
    Button(feedback_window, text="No", command=lambda: on_feedback_click(False, class_index, img, feedback_window)).pack(side='right', padx=10, pady=10)

def capture_image():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Image', frame)
        key = cv2.waitKey(1)
        if key == ord('c'):  # Press 'c' to capture the image
            img = frame.copy()
            break
        elif key == ord('q'):  # Press 'q' to quit without capturing
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    img = preprocess_image(img=img)
    class_index, confidence = classify_image(img)

    # Replace this line with the list of your ASL alphabet labels
    class_labels = ['A', 'B', 'C']  # You should update this list with all the ASL alphabet labels that your model can predict

    messagebox.showinfo("Result", f"Predicted Class: {class_labels[class_index]}\nConfidence: {confidence * 100:.2f}%")

    feedback_window = Toplevel(root)
    feedback_window.title("Feedback")
    Label(feedback_window, text="Was the prediction correct?", font=("Arial", 14)).pack(pady=10)

    Button(feedback_window, text="Yes", command=lambda: on_feedback_click(True, class_index, img, feedback_window)).pack(side='left', padx=10, pady=10)
    Button(feedback_window, text="No", command=lambda: on_feedback_click(False, class_index, img, feedback_window)).pack(side='right', padx=10, pady=10)

# Create GUI
root = ThemedTk(theme="equilux")  # You can choose from various themes available in ttkthemes
root.title("ASL Interpreter")

# Set window size and position
window_width, window_height = 400, 200
screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
position_x, position_y = (screen_width // 2) - (window_width // 2), (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

main_frame = Frame(root)
main_frame.pack(padx=10, pady=10)

label = Label(main_frame, text="ASL Interpreter", font=("Arial", 20, "bold"))
label.pack(pady=10, padx=80)

button_choose = Button(main_frame, text="Choose Image", style="TButton", command=on_button_click)
button_choose.pack(pady=10)

button_capture = Button(main_frame, text="Capture Image", style="TButton", command=capture_image)
button_capture.pack(pady=10)

root.mainloop()