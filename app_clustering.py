import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

# Select features for clustering (excluding student_id)
features = df.drop('student_id', axis=1)

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
            "Number of Clusters (K)",
            2, 10, 3,
            help="Number of clusters to form as well as the number of centroids to generate."
        )
    elif algorithm == "DBSCAN":
        eps = st.slider(
            "Epsilon (eps)",
            0.1, 2.0, 0.5, 0.1,
            help="The maximum distance between two samples for them to be considered as in the same neighborhood."
        )
        min_samples = st.slider(
            "Min Samples",
            2, 10, 5,
            help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point."
        )
    else:  # Hierarchical
        n_clusters_hierarchical = st.slider(
            "Number of Clusters",
            2, 10, 3,
            help="The number of clusters to find."
        )
        linkage = st.selectbox(
            "Linkage",
            ['ward', 'complete', 'average', 'single'],
            help="Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation."
        )


# Function to perform clustering and handle potential errors
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

        clusters = model.fit_predict(features)

        # Silhouette Score Calculation (only if more than one cluster)
        if len(set(clusters)) > 1:
            silhouette_avg = silhouette_score(features, clusters)
            db_index = davies_bouldin_score(features, clusters)
            ch_index = calinski_harabasz_score(features, clusters)
        else:
            silhouette_avg = None
            db_index = None
            ch_index = None

        return clusters, silhouette_avg, db_index, ch_index

    except Exception as e:
        st.error(f"An error occurred during clustering: {e}")
        return None, None, None, None


# Perform clustering based on the selected algorithm
clusters, silhouette_avg, db_index, ch_index = cluster_data(
    algorithm,
    n_clusters=n_clusters_kmeans if algorithm == "KMeans" else n_clusters_hierarchical,
    eps=eps if algorithm == "DBSCAN" else 0.5,
    min_samples=min_samples if algorithm == "DBSCAN" else 5,
    linkage=linkage if algorithm == "Hierarchical" else "ward",
)

if clusters is not None:
    df['cluster'] = clusters

    # Display the clustered data and silhouette score
    st.subheader(f"Clustered Data using {algorithm}:")
    st.dataframe(df)

    # Evaluation Metrics Section
    st.subheader("Clustering Evaluation Metrics")

    if silhouette_avg is not None:
        st.markdown(f"<h5>Silhouette Score: <span style='color:green;'>{silhouette_avg:.2f}</span></h5>",
                    unsafe_allow_html=True)
        st.markdown(f"<h5>Davies-Bouldin Index: <span style='color:green;'>{db_index:.2f}</span></h5>",
                    unsafe_allow_html=True)
        st.markdown(f"<h5>Calinski-Harabasz Index: <span style='color:green;'>{ch_index:.2f}</span></h5>",
                    unsafe_allow_html=True)
    else:
        st.warning("Evaluation metrics are not applicable. Only one cluster found.")

    # Visualization (example using two features)
    st.subheader("Cluster Visualization (Example)")
    fig, ax = plt.subplots()
    colors = ListedColormap(['red', 'green', 'blue', 'purple', 'orange'])
    scatter = ax.scatter(
        df['attendance_rate'], df['test_average'], c=df['cluster'], cmap=colors
    )
    ax.set_xlabel('Attendance Rate')
    ax.set_ylabel('Test Average')
    plt.title(f"Student Clusters ({algorithm})")

    # Dynamic Legend with Cluster Counts
    legend_labels = []
    for i in range(len(set(clusters))):
        cluster_count = list(clusters).count(i)
        legend_labels.append(f'Cluster {i} (n={cluster_count})')

    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    st.pyplot(fig)
