import os


# Install necessary OpenGL libraries on Linux
def install_opengl_libraries():
    os.system("sudo apt update")
    os.system("sudo apt install -y libgl1-mesa-glx")


# Check if running on Linux and install OpenGL libraries
if os.name == 'posix' and os.uname().sysname == 'Linux':
    print("Installing necessary OpenGL libraries...")
    install_opengl_libraries()
    print("OpenGL libraries installed successfully.")

# Now import the required modules
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
import random
import io
import uuid

# Suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load models
yolo_model = YOLO('./models/yolov8x.pt')
classification_model = load_model('./models/model.h5')


# Function to classify a cropped image
def classify_cropped_image(img_crop, model, target_size=(224, 224)):
    # Adjust target size to match model's expected input
    target_size = (model.input_shape[1], model.input_shape[2])
    img_resized = cv2.resize(img_crop, target_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = class_indices_inverse[np.argmax(prediction[0])]
    probability = np.max(prediction[0])
    return predicted_class, probability


# Function to extract frames from the video
def extract_random_frames(video_path, frames_number):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = sorted(random.sample(range(total_frames), frames_number))
    frames = []

    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


# Function to derive behavioral metrics
def derive_behavioral_metrics(predictions):
    engagement_classes = ["focused_mouth_closed", "focused_mouth_open", "listening", "raise_hand", "writing_reading"]
    distraction_classes = ["distracted_mouth_closed", "distracted_mouth_open", "using_smartphone"]
    fatigue_classes = ["fatigue", "sleeping"]

    total_students = len(predictions)
    engagement_count = sum(1 for pred in predictions if pred in engagement_classes)
    distraction_count = sum(1 for pred in predictions if pred in distraction_classes)
    fatigue_count = sum(1 for pred in predictions if pred in fatigue_classes)
    device_usage_count = sum(1 for pred in predictions if pred == "using_smartphone")

    engagement_rate = (engagement_count / total_students) * 100 if total_students else 0
    distraction_rate = (distraction_count / total_students) * 100 if total_students else 0
    fatigue_rate = (fatigue_count / total_students) * 100 if total_students else 0
    device_usage_rate = (device_usage_count / total_students) * 100 if total_students else 0

    return engagement_rate, distraction_rate, fatigue_rate, device_usage_rate


# Function to plot an array of images with a specified number of columns
def plot_images(images, cols=3, figsize=(15, 10)):
    rows = len(images) // cols + (1 if len(images) % cols != 0 else 0)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i, img in enumerate(images):
        row, col = divmod(i, cols)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if rows > 1:
            axes[row, col].imshow(img_rgb)
            axes[row, col].axis('off')
        else:
            axes[col].imshow(img_rgb)
            axes[col].axis('off')

    for j in range(len(images), rows * cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    return fig


# Load class labels
with open('./models/labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]
class_indices_inverse = {i: label for i, label in enumerate(class_labels)}

# Set page layout
st.set_page_config(page_title="Comprehensive Classification System", layout="wide")

# Header
st.title("Comprehensive Classification System for Online Learning")

# Navigation tabs
tabs = ["Normal Image", "Frames Extraction", "Real-Time Video"]
selected = option_menu(None, tabs, icons=["image", "film", "camera"], default_index=0, orientation="horizontal")

# Normal Image Tab
if selected == "Normal Image":
    st.header("Upload Normal Images")
    uploaded_file = st.file_uploader("Upload your file here (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.header("Uploaded Image")
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Convert to OpenCV format and ensure it's in RGB color space
        img = Image.open(uploaded_file)
        img = np.array(img.convert("RGB"))

        # Perform YOLO detection
        results = yolo_model(img)
        predictions = []

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
                predictions.append(predicted_class)
                label = f"{predicted_class}: {probability:.2f}"
            else:
                label = f"Other: {conf:.2f}"

            # Draw bounding box and label on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for bounding box
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green for label

        # No need to convert to RGB again, since it's already in RGB format
        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.axis('off')

        # Convert plot to a PNG image and display it in Streamlit
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
        img_buf.seek(0)
        st.header("Output Image")
        st.image(img_buf, caption="Output Image", use_column_width=True)

        # Calculate and display behavioral metrics
        engagement_rate, distraction_rate, fatigue_rate, device_usage_rate = derive_behavioral_metrics(predictions)

        st.header("Behavioral Metrics")
        st.markdown(f"Detected People: **<span style='color:green'>{len(predictions)}</span>**", unsafe_allow_html=True)
        st.markdown(f"Engagement Rate: **<span style='color:green'>{engagement_rate:.2f}%</span>**",
                    unsafe_allow_html=True)
        st.markdown(f"Distraction Rate: **<span style='color:green'>{distraction_rate:.2f}%</span>**",
                    unsafe_allow_html=True)
        st.markdown(f"Fatigue Rate: **<span style='color:green'>{fatigue_rate:.2f}%</span>**", unsafe_allow_html=True)
        st.markdown(f"Device Usage Rate: **<span style='color:green'>{device_usage_rate:.2f}%</span>**",
                    unsafe_allow_html=True)

        # Display the classification results
        st.header("Classification Results")
        for i, prediction in enumerate(predictions):
            st.markdown(f"Person {i + 1}: **{prediction}**")

# Frames Extraction Tab
if selected == "Frames Extraction":
    st.header("Upload Video for Frame Extraction")
    uploaded_video = st.file_uploader("Upload your video file here", type=["mp4", "mov", "avi"])
    frames_slider = st.slider("Number of frames to extract", min_value=1, max_value=20, value=5)

    if uploaded_video is not None:
        # Create output directory if it doesn't exist
        if not os.path.exists("output"):
            os.makedirs("output")

        # Unique UUID for the uploaded video
        video_path = f"output/temp_{str(uuid.uuid4())}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Create a progress bar
        progress_bar = st.progress(0)
        frames = extract_random_frames(video_path, frames_slider)
        detected_frames = []
        predictions = []

        for i, frame in enumerate(frames):
            results = yolo_model(frame)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = box.conf[0]

                if cls_id == 0:
                    img_crop = frame[y1:y2, x1:x2]
                    predicted_class, probability = classify_cropped_image(img_crop, classification_model)
                    predictions.append(predicted_class)
                    label = f"{predicted_class}: {probability:.2f}"
                else:
                    label = f"Other: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            detected_frames.append(frame)

            # Update progress bar
            progress_bar.progress((i + 1) / len(frames), text=f"Frame {i + 1} processed")

        fig = plot_images(detected_frames, cols=3)
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
        img_buf.seek(0)
        st.header("Extracted Frames with Detections")
        st.image(img_buf, caption="Extracted Frames with Detections", use_column_width=True)

        # Calculate and display behavioral metrics
        engagement_rate, distraction_rate, fatigue_rate, device_usage_rate = derive_behavioral_metrics(predictions)

        st.header("Behavioral Metrics")
        st.markdown(f"Engagement Rate: **<span style='color:green'>{engagement_rate:.2f}%</span>**",
                    unsafe_allow_html=True)
        st.markdown(f"Distraction Rate: **<span style='color:green'>{distraction_rate:.2f}%</span>**",
                    unsafe_allow_html=True)
        st.markdown(f"Fatigue Rate: **<span style='color:green'>{fatigue_rate:.2f}%</span>**", unsafe_allow_html=True)
        st.markdown(f"Device Usage Rate: **<span style='color:green'>{device_usage_rate:.2f}%</span>**",
                    unsafe_allow_html=True)

        # Display the classification results
        st.header("Classification Results")
        for i, prediction in enumerate(predictions):
            st.markdown(f"Person {i + 1}: **{prediction}**")

# Real-Time Video Tab
if selected == "Real-Time Video":
    st.header("Upload and Process Video in Real-Time")
    uploaded_video = st.file_uploader("Upload your video file here", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # Create output directory if it doesn't exist
        if not os.path.exists("output"):
            os.makedirs("output")

        video_path = f"output/temp_{str(uuid.uuid4())}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        stframe = st.empty()
        predictions = []

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = box.conf[0]

                if cls_id == 0:
                    img_crop = frame[y1:y2, x1:x2]
                    predicted_class, probability = classify_cropped_image(img_crop, classification_model)
                    predictions.append(predicted_class)
                    label = f"{predicted_class}: {probability:.2f}"
                else:
                    label = f"Other: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            stframe.image(frame, channels="BGR", use_column_width=True)

            progress_bar.progress((i + 1) / frame_count, text=f"Processing Frame {i + 1} of {frame_count}")

        cap.release()

        # Calculate and display behavioral metrics
        engagement_rate, distraction_rate, fatigue_rate, device_usage_rate = derive_behavioral_metrics(predictions)

        st.header("Behavioral Metrics")
        st.markdown(f"Engagement Rate: **<span style='color:green'>{engagement_rate:.2f}%</span>**",
                    unsafe_allow_html=True)
        st.markdown(f"Distraction Rate: **<span style='color:green'>{distraction_rate:.2f}%</span>**",
                    unsafe_allow_html=True)
        st.markdown(f"Fatigue Rate: **<span style='color:green'>{fatigue_rate:.2f}%</span>**", unsafe_allow_html=True)
        st.markdown(f"Device Usage Rate: **<span style='color:green'>{device_usage_rate:.2f}%</span>**",
                    unsafe_allow_html=True)

        # Display the classification results
        st.header("Classification Results")
        for i, prediction in enumerate(predictions):
            st.markdown(f"Person {i + 1}: **{prediction}**")
