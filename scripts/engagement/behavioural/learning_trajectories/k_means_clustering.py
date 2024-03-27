import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def calculate_k_means_clusters() -> None:
    data = {
        'learner_id': [1, 2, 3],
        'week_1': ['on track', 'behind', 'auditing'],
        'week_2': ['on track', 'auditing', 'out'],
    }
    df = pd.DataFrame(data)

    engagement_mapping = {'on track': 3, 'behind': 2, 'auditing': 1, 'out': 0}
    for col in df.columns[1:]:
        df[col] = df[col].map(engagement_mapping)

    X = df.drop(columns=['learner_id'])

    best_score = -1
    best_clusters = None

    for _ in range(100):
        kmeans = KMeans(n_clusters=4, random_state=None).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_clusters = kmeans.labels_

    df['cluster'] = best_clusters

    print(df)
    print(f"Best silhouette score: {best_score}")

    cluster_counts = df['cluster'].value_counts()
    print(cluster_counts)
