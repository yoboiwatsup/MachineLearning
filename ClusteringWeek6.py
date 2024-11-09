import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
retail_data = pd.read_excel('/mnt/data/Online Retail.xlsx')
clustering_data = pd.read_csv('/mnt/data/clusteringweek06.csv')

# Display the first few rows of each dataset to understand their structure
print("Online Retail Data:\n", retail_data.head())
print("\nClustering Data:\n", clustering_data.head())

# Preprocess the Data
# Online Retail data: selecting 'Quantity' and 'UnitPrice' as clustering features
retail_data = retail_data.dropna(subset=['Quantity', 'UnitPrice'])
retail_features = retail_data[['Quantity', 'UnitPrice']]
scaler = StandardScaler()
retail_scaled = scaler.fit_transform(retail_features)

# Clustering data: selecting numeric columns for clustering
clustering_data = clustering_data.dropna()
clustering_features = clustering_data.select_dtypes(include='number')
clustering_scaled = scaler.fit_transform(clustering_features)

# Define function for clustering and evaluation
def cluster_and_evaluate(data, model_name, model):
    # Fit the model
    labels = model.fit_predict(data)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(data, labels) if len(set(labels)) > 1 else -1
    print(f"{model_name} Silhouette Score: {silhouette_avg}")
    return labels, silhouette_avg

# Define clustering models
models = {
    'KMeans': KMeans(n_clusters=3, random_state=0),
    'Agglomerative': AgglomerativeClustering(n_clusters=3),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
}

# Run clustering on both datasets
results = {}
for name, model in models.items():
    print(f"\nClustering with {name} on Online Retail Data")
    retail_labels, retail_silhouette = cluster_and_evaluate(retail_scaled, name, model)
    
    print(f"\nClustering with {name} on Clustering Data")
    clustering_labels, clustering_silhouette = cluster_and_evaluate(clustering_scaled, name, model)
    
    # Store results for further analysis
    results[name] = {
        'retail_labels': retail_labels,
        'retail_silhouette': retail_silhouette,
        'clustering_labels': clustering_labels,
        'clustering_silhouette': clustering_silhouette
    }

# Visualize the Elbow Method for KMeans
def plot_elbow(data, max_k=10):
    distortions = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k+1), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()

# Plot Elbow for Online Retail Data
print("\nElbow Method for Online Retail Data")
plot_elbow(retail_scaled)

# Plot Elbow for Clustering Data
print("\nElbow Method for Clustering Data")
plot_elbow(clustering_scaled)

# Function for additional evaluation metrics
def evaluate_metrics(data, labels):
    # Davies-Bouldin Index
    db_index = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else -1
    # Calinski-Harabasz Index
    ch_index = calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else -1
    return db_index, ch_index

# Calculate and display additional metrics for each model on both datasets
for name, result in results.items():
    print(f"\nEvaluating {name} on Online Retail Data")
    db_index, ch_index = evaluate_metrics(retail_scaled, result['retail_labels'])
    print(f"Davies-Bouldin Index: {db_index}, Calinski-Harabasz Index: {ch_index}")
    
    print(f"Evaluating {name} on Clustering Data")
    db_index, ch_index = evaluate_metrics(clustering_scaled, result['clustering_labels'])
    print(f"Davies-Bouldin Index: {db_index}, Calinski-Harabasz Index: {ch_index}")

# Final Analysis to Determine the Best Model Based on Evaluation Metrics
best_models = {}
for dataset in ['retail', 'clustering']:
    print(f"\nSummary for {dataset.capitalize()} Data:")
    best_score = -1
    best_model = None
    for name, result in results.items():
        silhouette = result[f'{dataset}_silhouette']
        if silhouette > best_score:
            best_score = silhouette
            best_model = name
        print(f"{name}: Silhouette Score = {silhouette}")
    print(f"Best Model for {dataset.capitalize()} Data: {best_model} with Silhouette Score = {best_score}")
