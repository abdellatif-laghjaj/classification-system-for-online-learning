# Comprehensive Classification System for Online Learning

This project provides a comprehensive classification system designed for online learning environments. The system uses
advanced deep learning models for analyzing images and videos, extracting frames, and processing video data in
real-time. It can detect and classify various behaviors and states like engagement, distraction, and fatigue, providing
insights into student behavior.

## Features

- **Normal Image Processing:** Upload normal images to detect and classify behaviors.
- **Frame Extraction from Videos:** Extract and analyze frames from uploaded videos.
- **Real-Time Video Processing:** Process and classify uploaded videos in real-time.

## Technologies Used

- **Python:** Programming language used for implementing the system.
- **Streamlit:** Framework used for building the web application.
- **YOLO (You Only Look Once):** Object detection model for detecting objects in images and videos.
- **TensorFlow:** Framework used for loading and running the classification model.
- **OpenCV:** Library used for image and video processing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abdellatif-laghjaj/classification-system-for-online-learning.git
   ```
2. Install the required packages:
   ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit application:
    ```bash
    streamlit run app_classification.py --server.runOnSave true
    ```
4. Open the Streamlit app in your browser:
    ```
    http://localhost:8501
    ```

## Aknowledgements

- **Streamlit:** [https://streamlit.io/](https://streamlit.io/)
- **YOLO (You Only Look Once):** [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
- **TensorFlow:** [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **OpenCV:** [https://opencv.org/](https://opencv.org/)
- **Python:** [https://www.python.org/](https://www.python.org/)

## Author

- **Abdellatif Laghjaj** - [abdellatif-laghjaj](https://github.com/abdellatif-laghjaj)
