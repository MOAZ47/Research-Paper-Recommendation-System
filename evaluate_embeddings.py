import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# ---- Config ----
EMBEDDINGS_PATH = "paper_embeddings_ray.parquet"
METADATA_PATH = "papers.parquet"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---- Load Data ----
embeddings = pd.read_parquet(EMBEDDINGS_PATH)
metadata = pd.read_parquet(METADATA_PATH)[['id', 'categories']]

# Merge and handle duplicate category columns
df = embeddings.merge(metadata, on='id', suffixes=('', '_meta'))

# Select the correct categories column (priority to _meta if exists)
if 'categories_meta' in df.columns:
    df['categories'] = df['categories_meta']
elif 'categories' not in df.columns:
    raise KeyError("No categories column found after merge")

# Process categories - handle all possible formats
def process_categories(cats):
    if isinstance(cats, str):
        return [cats]
    elif isinstance(cats, list):
        return cats
    elif isinstance(cats, np.ndarray):
        return list(cats)
    return ['unknown']

df['categories'] = df['categories'].apply(process_categories)
df['primary_category'] = df['categories'].apply(lambda x: x[0] if len(x) > 0 else 'unknown')

# ---- Evaluation Metrics ----
def evaluate_embeddings(df, sample_size=1000):
    sample = df.sample(min(sample_size, len(df)))
    X = np.stack(sample['embedding'].values)
    
    distances = pairwise_distances(X, metric='cosine')
    np.fill_diagonal(distances, np.inf)
    
    category_map = sample['primary_category'].values
    results = []
    
    for k in [1, 3, 5, 10]:
        top_k = np.argpartition(distances, k, axis=1)[:, :k]
        correct = 0
        
        for i in range(len(sample)):
            query_cat = category_map[i]
            neighbor_cats = category_map[top_k[i]]
            correct += np.sum(neighbor_cats == query_cat)
        
        precision = correct / (len(sample) * k)
        results.append({'k': k, 'precision': precision})
    
    return pd.DataFrame(results)

# ---- Visualization ----
def visualize_embeddings(df, sample_size=500):
    sample = df.sample(min(sample_size, len(df)))
    X = np.stack(sample['embedding'].values)
    
    top_cats = Counter(df['primary_category']).most_common(5)
    cat_list = [cat for cat, _ in top_cats]
    mask = sample['primary_category'].isin(cat_list)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30)
    X_tsne = tsne.fit_transform(X[mask])
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=sample[mask]['primary_category'],
        palette='tab10', s=50, alpha=0.7
    )
    plt.title("t-SNE of Paper Embeddings (Top 5 Categories)")
    plt.savefig(f"{PLOTS_DIR}/tsne_visualization.png")
    plt.show()
    
    

if __name__ == "__main__":
    # Run Evaluation
    metrics_df = evaluate_embeddings(df)
    print("\n🔍 Embedding Quality Metrics:")
    print(metrics_df)
    
    # Generate Visualizations
    visualize_embeddings(df)
    
    # Save Results
    metrics_df.to_csv(f"{PLOTS_DIR}/embedding_metrics.csv", index=False)