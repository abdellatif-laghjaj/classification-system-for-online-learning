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
            n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)
        elif algorithm == "DBSCAN":
            eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 2, 10, 5)
        else:  # Hierarchical
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            linkage = st.selectbox("Linkage", ['ward', 'complete', 'average', 'single'])

            # Function to perform clustering and handle potential errors


    def cluster_data(algo_name):
        try:
            if algo_name == "KMeans":
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif algo_name == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
            else:  # Hierarchical
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

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


    # Perform clustering
    clusters, silhouette_avg = cluster_data(algorithm)

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
        colors = ListedColormap(['red', 'green', 'blue', 'purple', 'orange'])  # More colors for potential clusters
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

        # Benchmarking Visualization
        st.subheader("Algorithm Comparison (Silhouette Score)")
        algorithms = ["KMeans", "DBSCAN", "Hierarchical"]
        scores = []
        for algo in algorithms:
            _, score = cluster_data(algo)
            scores.append(score)

        fig, ax = plt.subplots()
        bars = ax.bar(algorithms, scores)
        ax.set_ylabel("Silhouette Score")
        plt.title("Clustering Algorithm Performance")

        # Add score labels on top of the bars (if score is available)
        for bar, score in zip(bars, scores):
            if score is not None:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f'{score:.2f}', ha='center', va='bottom')
            else:
                st.warning(f"Silhouette Score for {algo} is not available.")

        st.pyplot(fig)
