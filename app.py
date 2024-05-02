import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import logging
import io

# Suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load models
yolo_model = YOLO('./models/yolov8x.pt')
classification_model = load_model('./models/model.h5')


# Function to classify a cropped image
def classify_cropped_image(img_crop, model, target_size=(224, 224)):
    img_resized = cv2.resize(img_crop, target_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    prediction = model.predict(img_array)
    predicted_class = class_indices_inverse[np.argmax(prediction[0])]
    probability = np.max(prediction[0])
    return predicted_class, probability


# Load class labels
with open('./models/labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]
class_indices_inverse = {i: label for i, label in enumerate(class_labels)}

# Set page layout
st.set_page_config(page_title="Comprehensive Classification System", layout="wide")

# Header
st.title("Comprehensive Classification System for Online Learning")

# Navigation tabs
tabs = ["Normal Image", "Frames Extraction", "Raw Video"]
selected = option_menu(None, tabs, icons=["image", "film", "camera"], default_index=0, orientation="horizontal")

# Main content
if selected == "Normal Image":
    st.header("Upload Normal Images")
    uploaded_file = st.file_uploader("Upload your file here (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Convert to OpenCV format
        img = Image.open(uploaded_file)
        img = np.array(img)

        # Perform YOLO detection
        results = yolo_model(img)

        # Initialize a figure for plotting
        fig, ax = plt.subplots()

        # Plot the original image with detected boxes and labels
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = box.conf[0]

            # Check if the detected class is 'person'
            if cls_id == 0:
                img_crop = img[y1:y2, x1:x2]
                predicted_class, probability = classify_cropped_image(img_crop, classification_model)
                label = f"{predicted_class}: {probability:.2f}"
            else:
                label = f"Other: {conf:.2f}"

            # Draw bounding box and label on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # Convert plot to a PNG image and display it in Streamlit
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        st.image(img_buf, caption="Detected Image", use_column_width=True)

        # Display behavioral metrics (replace with actual metrics calculation)
        st.write(f"Detected People: {len(results[0].boxes)}")
        st.write("Behavioral Metrics:")
        st.write(f"Engagement Rate: {np.random.uniform(0, 100):.2f}%")
        st.write(f"Distraction Rate: {np.random.uniform(0, 100):.2f}%")
        st.write(f"Fatigue Rate: {np.random.uniform(0, 100):.2f}%")
        st.write(f"Device Usage Rate: {np.random.uniform(0, 100):.2f}%")

# Frames Extraction Tab
elif selected == "Frames Extraction":
    st.header("Upload Video for Frame Extraction")
    uploaded_video = st.file_uploader("Upload your video file here", type=["mp4", "mov", "avi"])
    frames_slider = st.slider("Number of frames to extract", min_value=1, max_value=20, value=5)
    if uploaded_video is not None:
        st.video(uploaded_video)
        # Placeholder for frames extraction result
        st.write("Frames will be shown here")

# Raw Video Tab
elif selected == "Raw Video":
    st.header("Upload Raw Video")
    uploaded_video = st.file_uploader("Upload your raw video file here", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        st.video(uploaded_video)
        # Placeholder for classification result
        st.write("Results will be shown here")
