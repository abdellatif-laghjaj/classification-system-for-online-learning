import streamlit as st
from streamlit_option_menu import option_menu

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
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        # Placeholder for classification result
        st.write("Results will be shown here")

elif selected == "Frames Extraction":
    st.header("Upload Video for Frame Extraction")
    uploaded_video = st.file_uploader("Upload your video file here", type=["mp4", "mov", "avi"])
    frames_slider = st.slider("Number of frames to extract", min_value=1, max_value=20, value=5)
    if uploaded_video is not None:
        st.video(uploaded_video)
        # Placeholder for frames extraction result
        st.write("Frames will be shown here")

elif selected == "Raw Video":
    st.header("Upload Raw Video")
    uploaded_video = st.file_uploader("Upload your raw video file here", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        st.video(uploaded_video)
        # Placeholder for classification result
        st.write("Results will be shown here")