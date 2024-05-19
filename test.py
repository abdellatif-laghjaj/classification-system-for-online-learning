import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import logging
import uuid
import random

# Suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set page layout
st.set_page_config(page_title="Image Classification System", layout="wide")

# Class labels
class_labels = [
    "distracted_mouth_closed",
    "distracted_mouth_open",
    "fatigue",
    "focused_mouth_closed",
    "focused_mouth_open",
    "listening",
    "raise_hand",
    "sleeping",
    "using_smartphone",
    "writing_reading",
]


# Function to extract features using VGG16
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()


# Function to build a CNN model for feature extraction
def build_cnn_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(7 * 7 * 512,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to classify features using the selected classifier
def classify_features(X_train, y_train, X_test, y_test, classifier_name):
    if classifier_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif classifier_name == 'Naive Bayes':
        model = GaussianNB()
    elif classifier_name == 'k-NN':
        model = KNeighborsClassifier(n_neighbors=3)
    elif classifier_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif classifier_name == 'SVM':
        model = SVC()

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    unique_classes = np.unique(y_test)
    target_names = [class_labels[i] for i in unique_classes]

    report = classification_report(y_test, pred, target_names=target_names)

    return accuracy, report


# Header
st.title("Image Classification System")

# Upload image file
uploaded_file = st.file_uploader("Upload your image file here (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    img_path = f"temp_{uuid.uuid4()}.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load VGG16 model for feature extraction
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    feature_model = Model(inputs=base_model.input, outputs=x)

    # Extract features
    features = extract_features(img_path, feature_model)

    # Load or generate dataset
    # For demonstration, let's use random data (replace with actual data loading)
    X = np.random.rand(100, 512)  # 100 samples, 512 features each
    y = np.random.randint(0, len(class_labels), 100)  # 100 labels

    # Append the uploaded image features to the dataset
    X = np.vstack([X, features])
    y = np.append(y, random.randint(0, len(class_labels) - 1))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add a selection box for the classifier
    classifier_name = st.selectbox('Select Classifier',
                                   ('Logistic Regression', 'Naive Bayes', 'k-NN', 'Decision Tree', 'SVM'))

    # Classify using the selected classifier
    accuracy, report = classify_features(X_train, y_train, X_test, y_test, classifier_name)

    # Display the results
    st.write(f"{classifier_name} Accuracy: {accuracy:.2f}")
    st.text(report)

    # Display uploaded image
    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    # Clean up temporary file
    os.remove(img_path)
