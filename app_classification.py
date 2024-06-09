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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import pandas as pd

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
st.set_page_config(page_title="Comprehensive Student Behavior Analysis", layout="wide")
st.markdown(
    "<h3 style='text-align: center;'>Comprehensive "
    "<span style='color: red'>Object Detection</span>, "
    "<span style='color: red'>Classification</span>, "
    "and <span style='color: red'>Clustering</span> System</h3>",
    unsafe_allow_html=True
)

# Sidebar for selecting modes
st.sidebar.title("Options")
st.sidebar.write("Choose between Behavior Analysis or Clustering.")
mode = st.sidebar.selectbox("Select an option", ["Behavior Analysis", "Clustering"])

# Navigation tabs for behavior analysis
if mode == "Behavior Analysis":
    tabs = ["Normal Image", "Frames Extraction", "Real-Time Video"]
    selected = option_menu(None, tabs, icons=["image", "film", "camera"], default_index=0, orientation="horizontal")
    st.sidebar.header("Model Settings")

    # Load YOLO model for object detection
    yolo_model_path = st.sidebar.text_input("YOLO Model Path (leave empty for default)", "./models/yolov8x.pt")
    if not yolo_model_path:
        yolo_model_path = "./models/yolov8x.pt"
    yolo_model = YOLO(yolo_model_path)

    # Load classification model
    classification_model_path = st.sidebar.text_input("Classification Model Path (leave empty for default)",
                                                      "./models/model.h5")
    if not classification_model_path:
        classification_model_path = "./models/model.h5"
    classification_model = load_model(classification_model_path)

    # Load class labels
    labels_path = st.sidebar.text_input("Labels File Path (leave empty for default)", "./models/labels.txt")
    if not labels_path:
        labels_path = "./models/labels.txt"
    with open(labels_path, 'r') as f:
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
        engagement_classes = ["focused_mouth_closed", "focused_mouth_open", "listening", "raise_hand",
                              "writing_reading"]
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


    # Function to process uploaded image
    def process_uploaded_image(uploaded_file, option):
        if uploaded_file is not None:
            # Read the image file
            img = Image.open(uploaded_file)
            image_np = np.array(img)
            st.image(img, caption='Uploaded Image', use_column_width=True)

            # Process the image with YOLO model
            results = yolo_model(image_np)

            img_crops = []
            predictions = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = box.conf[0]

                if cls_id == 0:  # Assuming cls_id == 0 is for person
                    img_crop = image_np[y1:y2, x1:x2]
                    img_crops.append(img_crop)
                    if option == "Classification":
                        predicted_class, probability = classify_cropped_image(img_crop, classification_model)
                        predictions.append(predicted_class)
                        label = f"{predicted_class}: {probability:.2f}"
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


    # Function to process uploaded video
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
                engagement_rate, distraction_rate, fatigue_rate, device_usage_rate = derive_behavioral_metrics(
                    predictions)
                st.markdown(f"Engagement Rate: {engagement_rate:.2f}%")
                st.markdown(f"Distraction Rate: {distraction_rate:.2f}%")
                st.markdown(f"Fatigue Rate: {fatigue_rate:.2f}%")
                st.markdown(f"Device Usage Rate: {device_usage_rate:.2f}%")


    # Function to process real-time video
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
                        label = f"Other: {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                stframe.image(frame, channels="BGR", use_column_width=True)
                progress_bar.progress((i + 1) / frame_count, text=f"Processing Frame {i + 1} of {frame_count}")

            cap.release()

            if option == "Classification":
                # Display the classification results
                st.header("Classification Results")
                engagement_rate, distraction_rate, fatigue_rate, device_usage_rate = derive_behavioral_metrics(
                    predictions)
                st.markdown(f"Engagement Rate: {engagement_rate:.2f}%")
                st.markdown(f"Distraction Rate: {distraction_rate:.2f}%")
                st.markdown(f"Fatigue Rate: {fatigue_rate:.2f}%")
                st.markdown(f"Device Usage Rate: {device_usage_rate:.2f}%")


    # Main app logic for behavior analysis
    if mode == "Behavior Analysis":
        if selected == "Normal Image":
            st.header("Upload Normal Images")
            uploaded_file = st.file_uploader("Upload your file here (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
            process_uploaded_image(uploaded_file, selected)

        elif selected == "Frames Extraction":
            st.header("Upload Video for Frame Extraction")
            uploaded_video = st.file_uploader("Upload your video file here", type=["mp4", "mov", "avi"])
            frames_slider = st.slider("Select the number of frames to extract", min_value=1, max_value=50, value=10)
            process_uploaded_video(uploaded_video, frames_slider, selected)

        elif selected == "Real-Time Video":
            st.header("Upload and Process Video in Real-Time")
            uploaded_video = st.file_uploader("Upload your video file here", type=["mp4", "mov", "avi"])
            process_real_time_video(uploaded_video, selected)

# Clustering Section
elif mode == "Clustering":
    st.title("Student Behavior Clustering 📊✨")
    st.write("This app performs clustering on student behavior data to identify patterns and segments of students.")

    # Two option menus: App, About
    tabs = ["App", "About"]
    app_mode = option_menu(None, options=tabs, icons=["📊", "❓"], default_index=0, orientation="horizontal")

    # --- Sidebar for Settings and File Upload ---
    st.sidebar.header("Data and Clustering Settings")

    # File Upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file or use default:", type=["csv"]
    )

    # Use a default dataset if no file is uploaded
    if uploaded_file is None:
        df = pd.read_csv("clustering_data.csv")
    else:
        df = pd.read_csv(uploaded_file)

    # --- Data Preprocessing (Example: Handling Missing Values) ---
    # Replace this with your specific data cleaning needs
    df.fillna(df.mean(), inplace=True)

    # --- Feature Engineering (Example) ---
    df['engagement_score'] = (
            df['attendance_rate'] * 0.5 +
            df['test_average'] * 0.5
    )

    # Select features for clustering
    features = df[['attendance_rate', 'test_average', 'engagement_score']]

    # Standard Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Sidebar for Algorithm Selection and Parameter Tuning
    st.sidebar.header("Clustering Settings")
    algorithm = st.sidebar.selectbox(
        "Select Algorithm:",
        ("KMeans", "DBSCAN", "Hierarchical")
    )

    # Default values for parameters
    n_clusters_kmeans = 3
    eps = 0.5
    min_samples = 5
    n_clusters_hierarchical = 3
    linkage = 'ward'

    # Parameter tuning section
    with st.sidebar.expander("Algorithm Parameters"):
        if algorithm == "KMeans":
            n_clusters_kmeans = st.slider(
                "Number of Clusters (K)", 2, 10, 3,
                help="Number of clusters to form for KMeans."
            )
        elif algorithm == "DBSCAN":
            eps = st.slider(
                "Epsilon (eps)", 0.1, 2.0, 0.5, 0.1,
                help="Maximum distance between two samples for one to be considered as in the neighborhood of the other for DBSCAN."
            )
            min_samples = st.slider(
                "Min Samples", 2, 10, 5,
                help="The number of samples in a neighborhood for a point to be considered as a core point for DBSCAN."
            )
        else:  # Hierarchical
            n_clusters_hierarchical = st.slider(
                "Number of Clusters", 2, 10, 3,
                help="Number of clusters to find for hierarchical clustering."
            )
            linkage = st.selectbox(
                "Linkage", ['ward', 'complete', 'average', 'single'],
                help="Which linkage criterion to use for hierarchical clustering."
            )


    # Function to perform clustering
    def cluster_data(algo_name, **kwargs):
        try:
            if algo_name == "KMeans":
                model = KMeans(n_clusters=kwargs.get('n_clusters', 3), random_state=42)
            elif algo_name == "DBSCAN":
                model = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
            else:  # Hierarchical
                model = AgglomerativeClustering(
                    n_clusters=kwargs.get('n_clusters', 3),
                    linkage=kwargs.get('linkage', 'ward')
                )

            clusters = model.fit_predict(scaled_features)
            return clusters

        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")
            return None


    # Perform clustering
    clusters = cluster_data(
        algorithm,
        n_clusters=n_clusters_kmeans if algorithm == "KMeans" else n_clusters_hierarchical,
        eps=eps if algorithm == "DBSCAN" else 0.5,
        min_samples=min_samples if algorithm == "DBSCAN" else 5,
        linkage=linkage if algorithm == "Hierarchical" else "ward",
    )

    # THE APP CONTENT
    if app_mode == "About":
        st.write(
            """
            ## About
            This app performs clustering on student behavior data to identify patterns and segments of students.

            ### Data
            The dataset contains student information such as attendance rate, test average, and engagement score.

            ### Clustering Algorithms
            - **KMeans:** Partitions data into K clusters based on feature similarity.
            - **DBSCAN:** Density-based clustering to identify outliers and clusters of varying shapes.
            - **Hierarchical:** Builds a tree of clusters to identify subgroups.

            ### Evaluation Metrics
            - **Silhouette Score:** Measures how similar an object is to its cluster compared to other clusters.
            - **Davies-Bouldin Index:** Computes the average similarity between each cluster and its most similar one.
            - **Calinski-Harabasz Index:** Ratio of the sum of between-clusters dispersion and within-cluster dispersion.

            ### Cluster Profiling
            - Parallel coordinates plot to visualize and compare clusters across multiple features.

            ### Interpretation of Clusters
            - Provides insights into each cluster based on the average values of features.
            """
        )
        st.write(
            """
            ## How to Use
            1. **Upload Data:** Upload your own CSV file or use the default dataset.
            2. **Select Algorithm:** Choose between KMeans, DBSCAN, and Hierarchical clustering.
            3. **Set Parameters:** Adjust the clustering parameters in the sidebar.
            4. **Interpret Results:** Explore the clustered data, evaluation metrics, and cluster profiles.
            """
        )
        st.write(
            """
            ## Contact
            If you have any questions or feedback, feel free to connect with me on:
            - [LinkedIn](https://www.linkedin.com/in/abdellatif-laghjaj)
            - [GitHub](https://www.github.com/abdellatif-laghjaj)
            """
        )

    elif app_mode == "App":
        if clusters is not None:
            df['cluster'] = clusters

            # --- Display Clustered Data ---
            st.subheader(f"Clustered Data using {algorithm}:")
            st.dataframe(df)

            # --- Evaluation Metrics ---
            if len(set(clusters)) > 1:
                silhouette_avg = silhouette_score(scaled_features, clusters)
                db_index = davies_bouldin_score(scaled_features, clusters)
                ch_index = calinski_harabasz_score(scaled_features, clusters)

                st.subheader("Clustering Evaluation Metrics")
                st.markdown(f"**Silhouette Score:** {silhouette_avg:.2f}", unsafe_allow_html=True)
                st.markdown(f"**Davies-Bouldin Index:** {db_index:.2f}", unsafe_allow_html=True)
                st.markdown(f"**Calinski-Harabasz Index:** {ch_index:.2f}", unsafe_allow_html=True)
            else:
                st.warning("Evaluation metrics are not applicable. Only one cluster found.")

            # --- Interactive 3D Scatter Plot with Plotly ---
            st.subheader("Interactive 3D Cluster Visualization")
            fig = px.scatter_3d(
                df,
                x='attendance_rate',
                y='test_average',
                z='engagement_score',
                color='cluster',
                title=f"Student Clusters ({algorithm})",
                labels={'attendance_rate': 'Attendance Rate',
                        'test_average': 'Test Average',
                        'engagement_score': 'Engagement Score'}
            )
            st.plotly_chart(fig)

            # --- Cluster Profiling (Example using Plotly) ---
            st.subheader("Cluster Profile Visualization")
            st.write(
                "The parallel coordinates plot is a way to visualize and compare clusters across multiple features.")
            profile_features = ['attendance_rate', 'test_average', 'engagement_score']
            cluster_means = df.groupby('cluster')[profile_features].mean().reset_index()

            fig_profile = px.parallel_coordinates(
                cluster_means,
                color='cluster',
                dimensions=profile_features,
                title="Parallel Coordinates Plot for Cluster Profiles"
            )
            st.plotly_chart(fig_profile)

            # --- Dynamic Interpretation of Clusters ---
            st.subheader("Interpretation of Clusters")
            for cluster_num in cluster_means['cluster']:
                cluster_data = cluster_means[cluster_means['cluster'] == cluster_num]
                st.write(f"**Cluster {cluster_num}:**")
                for feature in profile_features:
                    st.write(f"- **{feature.replace('_', ' ').title()}:** {cluster_data[feature].values[0]:.2f}")

                highest_feature = cluster_data[profile_features].idxmax(axis=1).values[0]
                lowest_feature = cluster_data[profile_features].idxmin(axis=1).values[0]

                st.write(f"This cluster has the highest average {highest_feature.replace('_', ' ')} "
                         f"and the lowest average {lowest_feature.replace('_', ' ')}.")
                st.write("---")

            # Additional insights based on cluster characteristics can be added here.
        else:
            st.warning("Please configure the clustering settings and run the algorithm first.")
