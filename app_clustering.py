import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Student Behavior Clustering")

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
    st.write("The parallel coordinates plot is a way to visualize and compare clusters across multiple features.")
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
    st.error("An error occurred during clustering. Please check the algorithm parameters.")
