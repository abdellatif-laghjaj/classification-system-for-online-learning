import os
import streamlit as st
from matplotlib import pyplot as plt
from streamlit_option_menu import option_menu
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import logging
import random
import io
import uuid
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# Ensure necessary OpenGL libraries are installed on Linux
def install_opengl_libraries():
    os.system("sudo apt update")
    os.system("sudo apt install -y libgl1-mesa-glx")


if os.name == 'posix' and os.uname().sysname == 'Linux':
    print("Installing necessary OpenGL libraries...")
    install_opengl_libraries()
    print("OpenGL libraries installed successfully.")

# Set up Streamlit app layout
st.set_page_config(page_title="Comprehensive Classification System", layout="wide")
st.markdown(
    "<h3 style='text-align: center;'>Comprehensive "
    "<span style='color: red'>Object Detection</span>, "
    "<span style='color: red'>Classification</span>, "
    "and <span style='color: red'>Clustering</span> System</h3>",
    unsafe_allow_html=True
)

# Sidebar for selecting Classification or Clustering
st.sidebar.title("Options")
st.sidebar.write("You can choose between Classification and Clustering.")
st.sidebar.success("Classification: Classify detected objects.")
st.sidebar.success("Clustering: Cluster detected objects.")
option = st.sidebar.selectbox("Select an option", ["Clustering", "Classification"])

# Additional clustering algorithm selection
if option == "Clustering":
    clustering_algorithm = st.sidebar.selectbox("Select Clustering Algorithm",
                                                ["K-Means", "KNN", "DBSCAN", "Hierarchical Clustering"])

# Navigation tabs
tabs = ["Normal Image", "Frames Extraction", "Real-Time Video"]
selected = option_menu(None, tabs, icons=["image", "film", "camera"], default_index=0, orientation="horizontal")

# Load YOLO model for object detection
yolo_model = YOLO('./models/yolov8x.pt')

# Load classification model
classification_model = load_model('./models/model.h5')

# Load class labels
with open('./models/labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]
class_indices_inverse = {i: label for i, label in enumerate(class_labels)}


# Function to classify a cropped image
def classify_cropped_image(img_crop, model, target_size=(224, 224)):
    target_size = (model.input_shape[1], model.input_shape[2])
    img_resized = cv2.resize(img_crop, target_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = class_indices_inverse[np.argmax(prediction[0])]
    probability = np.max(prediction[0])
    return predicted_class, probability


# Function to extract random frames from video
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


# Function to plot images in a grid
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


# Function to cluster images
def cluster_images(images, algorithm, max_clusters=3):
    if len(images) == 0:
        return []

    # Convert images to feature vectors
    feature_vectors = []
    for img in images:
        img_resized = cv2.resize(img, (224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = img_array.flatten()
        feature_vectors.append(img_array)

    feature_vectors = np.array(feature_vectors)

    # Standardize the features
    scaler = StandardScaler()
    feature_vectors = scaler.fit_transform(feature_vectors)

    # Apply PCA for dimensionality reduction
    n_components = min(len(images), 50)  # Ensure n_components does not exceed the number of samples
    pca = PCA(n_components=n_components)
    feature_vectors_pca = pca.fit_transform(feature_vectors)

    # Adjust number of clusters to be less than or equal to the number of samples
    n_clusters = min(len(images), max_clusters)

    # Apply selected clustering algorithm
    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "KNN":
        model = NearestNeighbors(n_neighbors=n_clusters)
        model.fit(feature_vectors_pca)
        return model.kneighbors(return_distance=False)[:, 0]
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=0.5, min_samples=5)
    elif algorithm == "Hierarchical Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError("Unknown clustering algorithm: " + algorithm)

    labels = model.fit_predict(feature_vectors_pca)
    return labels


def process_uploaded_image(uploaded_file, option):
    if uploaded_file is not None:
        # Read the image file
        img = Image.open(uploaded_file)
        image_np = np.array(img)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Process the image with YOLO model
        results = yolo_model(image_np)

        img_crops = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = box.conf[0]

            if cls_id == 0:  # Assuming cls_id == 0 is for person
                img_crop = image_np[y1:y2, x1:x2]
                img_crops.append(img_crop)
                if option == "Classification":
                    predicted_class, probability = classify_cropped_image(img_crop, classification_model)
                    label = f"{predicted_class}: {probability:.2f}"
                else:
                    label = f"Person: {conf:.2f}"
            else:
                label = f"Other: {conf:.2f}"

            # Draw bounding box and label on the image
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        st.image(image_np, caption='Processed Image', use_column_width=True)

        if option == "Classification":
            st.header("Classification Results")
            for i, img_crop in enumerate(img_crops):
                predicted_class, probability = classify_cropped_image(img_crop, classification_model)
                st.markdown(f"Person {i + 1}: **{predicted_class}** ({probability:.2f})")
        else:
            if img_crops:
                st.header("Clustering Results")
                st.subheader(f"Using {clustering_algorithm} Algorithm")
                labels = cluster_images(img_crops, clustering_algorithm)
                cluster_dict = {}
                for i, label in enumerate(labels):
                    cluster_dict.setdefault(label, []).append(img_crops[i])

                for cluster_id, cluster_imgs in cluster_dict.items():
                    st.markdown(f"### Cluster {cluster_id}")
                    fig = plot_images(cluster_imgs, cols=3)
                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
                    img_buf.seek(0)
                    st.image(img_buf, caption=f"Cluster {cluster_id} Images", use_column_width=True)

                # Interactive cluster exploration
                selected_cluster = st.selectbox("Select a cluster to view images", sorted(cluster_dict.keys()))
                st.markdown(f"### Images in Cluster {selected_cluster}")
                fig = plot_images(cluster_dict[selected_cluster], cols=3)
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
                img_buf.seek(0)
                st.image(img_buf, caption=f"Images in Cluster {selected_cluster}", use_column_width=True)
            else:
                st.warning("No crops found for clustering. Please upload another image.")


def process_uploaded_video(uploaded_video, frames_slider, option):
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
                    if option == "Classification":
                        predicted_class, probability = classify_cropped_image(img_crop, classification_model)
                        predictions.append(predicted_class)
                        label = f"{predicted_class}: {probability:.2f}"
                    else:
                        label = f"Person: {conf:.2f}"
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

        if option == "Classification":
            # Display the classification results
            st.header("Classification Results")
            for i, prediction in enumerate(predictions):
                st.markdown(f"Person {i + 1}: **{prediction}**")
        else:
            st.header("Clustering Results")
            st.subheader(f"Using {clustering_algorithm} Algorithm")
            # Cluster the images
            img_crops = [frame[y1:y2, x1:x2] for frame in detected_frames]
            labels = cluster_images(img_crops, clustering_algorithm)
            cluster_dict = {}
            for i, label in enumerate(labels):
                cluster_dict.setdefault(label, []).append(img_crops[i])

            for cluster_id, cluster_images in cluster_dict.items():
                st.markdown(f"### Cluster {cluster_id}")
                fig = plot_images(cluster_images, cols=3)
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
                img_buf.seek(0)
                st.image(img_buf, caption=f"Cluster {cluster_id} Images", use_column_width=True)

            # Interactive cluster exploration
            selected_cluster = st.selectbox("Select a cluster to view images", sorted(cluster_dict.keys()))
            st.markdown(f"### Images in Cluster {selected_cluster}")
            fig = plot_images(cluster_dict[selected_cluster], cols=3)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
            img_buf.seek(0)
            st.image(img_buf, caption=f"Images in Cluster {selected_cluster}", use_column_width=True)


def process_real_time_video(uploaded_video, option):
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
        detected_frames = []

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
                    detected_frames.append(img_crop)
                    if option == "Classification":
                        predicted_class, probability = classify_cropped_image(img_crop, classification_model)
                        predictions.append(predicted_class)
                        label = f"{predicted_class}: {probability:.2f}"
                    else:
                        label = f"Person: {conf:.2f}"
                else:
                    label = f"Other: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            stframe.image(frame, channels="BGR", use_column_width=True)
            progress_bar.progress((i + 1) / frame_count, text=f"Processing Frame {i + 1} of {frame_count}")

        cap.release()

        if option == "Classification":
            # Display the classification results
            st.header("Classification Results")
            for i, prediction in enumerate(predictions):
                st.markdown(f"Person {i + 1}: **{prediction}**")
        else:
            st.header("Clustering Results")
            st.subheader(f"Using {clustering_algorithm} Algorithm")
            # Cluster the images
            labels = cluster_images(detected_frames, clustering_algorithm)
            cluster_dict = {}
            for i, label in enumerate(labels):
                cluster_dict.setdefault(label, []).append(detected_frames[i])

            for cluster_id, cluster_images in cluster_dict.items():
                st.markdown(f"### Cluster {cluster_id}")
                fig = plot_images(cluster_images, cols=3)
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
                img_buf.seek(0)
                st.image(img_buf, caption=f"Cluster {cluster_id} Images", use_column_width=True)

            # Interactive cluster exploration
            selected_cluster = st.selectbox("Select a cluster to view images", sorted(cluster_dict.keys()))
            st.markdown(f"### Images in Cluster {selected_cluster}")
            fig = plot_images(cluster_dict[selected_cluster], cols=3)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
            img_buf.seek(0)
            st.image(img_buf, caption=f"Images in Cluster {selected_cluster}", use_column_width=True)


# Main app logic
if selected == "Normal Image":
    st.header("Upload Normal Images")
    uploaded_file = st.file_uploader("Upload your file here (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    process_uploaded_image(uploaded_file, option)

elif selected == "Frames Extraction":
    st.header("Upload Video for Frame Extraction")
    uploaded_video = st.file_uploader("Upload your video file here", type=["mp4", "mov", "avi"])
    frames_slider = st.slider("Select the number of frames to extract", min_value=1, max_value=50, value=10)
    process_uploaded_video(uploaded_video, frames_slider, option)

elif selected == "Real-Time Video":
    st.header("Upload and Process Video in Real-Time")
    uploaded_video = st.file_uploader("Upload your video file here", type=["mp4", "mov", "avi"])
    process_real_time_video(uploaded_video, option)
