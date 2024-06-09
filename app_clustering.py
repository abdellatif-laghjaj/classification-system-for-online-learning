import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

st.title("Student Behavior Clustering")
st.write("Upload a CSV file containing student data to perform clustering.")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file...", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Select features for clustering (excluding student_id)
    features = df.drop('student_id', axis=1)

    # Perform KMeans clustering (e.g., with 3 clusters)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features)

    # Add cluster labels to the DataFrame
    df['cluster'] = clusters

    # Display the clustered data
    st.subheader("Clustered Data:")
    st.dataframe(df)

    # Visualization (example using two features)
    st.subheader("Cluster Visualization (Example)")
    fig, ax = plt.subplots()
    # Scatter plot with custom colors for each cluster
    colors = ListedColormap(['red', 'green', 'blue'])  # You can customize these colors
    scatter = ax.scatter(df['attendance_rate'], df['test_average'], c=df['cluster'], cmap=colors)
    ax.set_xlabel('Attendance Rate')
    ax.set_ylabel('Test Average')
    plt.title("Student Clusters")

    # Add a legend for cluster labels
    legend_labels = [f'Cluster {i}' for i in range(kmeans.n_clusters)]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    st.pyplot(fig)
