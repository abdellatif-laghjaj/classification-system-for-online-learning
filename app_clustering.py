import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
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

    # Sidebar for Algorithm Selection and Parameter Tuning
    st.sidebar.header("Clustering Settings")
    algorithm = st.sidebar.selectbox(
        "Select Algorithm:",
        ("KMeans", "DBSCAN", "Hierarchical")
    )

    # Parameter tuning section (initially hidden)
    with st.sidebar.expander("Algorithm Parameters"):
        if algorithm == "KMeans":
            n_clusters_kmeans = st.slider("Number of Clusters (K)", 2, 10, 3)
        elif algorithm == "DBSCAN":
            eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 2, 10, 5)
        else:  # Hierarchical
            n_clusters_hierarchical = st.slider("Number of Clusters", 2, 10, 3)
            linkage = st.selectbox("Linkage", ['ward', 'complete', 'average', 'single'])


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
            else:
                silhouette_avg = None  # Indicate score not applicable

            return clusters, silhouette_avg

        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")
            return None, None


    # Perform clustering based on the selected algorithm
    clusters, silhouette_avg = cluster_data(
        algorithm,
        n_clusters=n_clusters_kmeans if algorithm == "KMeans" else 3,
        eps=eps if algorithm == "DBSCAN" else 0.5,
        min_samples=min_samples if algorithm == "DBSCAN" else 5,
        linkage=linkage if algorithm == "Hierarchical" else "ward",
    )

    if clusters is not None:
        df['cluster'] = clusters

        # Display the clustered data and silhouette score
        st.subheader(f"Clustered Data using {algorithm}:")
        st.dataframe(df)
        if silhouette_avg is not None:
            st.write(f"Silhouette Score: {silhouette_avg:.2f}")
        else:
            st.warning(
                "Silhouette Score is not applicable. Only one cluster found."
            )

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
