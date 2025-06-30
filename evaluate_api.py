import requests
import pandas as pd
import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# ---- Config ----
API_URL = "http://localhost:8000/search"
K_VALUES = [1, 3, 5, 10]  # Evaluate at multiple K values
METADATA_PATH = "faiss_index/metadata.parquet"
RESULTS_DIR = "logs"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Load Metadata ----
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")

metadata = pd.read_parquet(METADATA_PATH)
metadata['categories'] = metadata['categories'].apply(lambda x: [x] if isinstance(x, str) else x)

# ---- Benchmark Queries ----
queries = [
    ("Graph Neural Networks for Chemistry", ["cs.LG", "physics.chem-ph"]),
    ("Transformers in NLP", ["cs.CL", "cs.AI"]),
    ("Reinforcement Learning for Robotics", ["cs.RO", "cs.LG"]),
    ("Quantum Computing and Algorithms", ["quant-ph", "cs.DS"]),
    ("Deep Learning for Radiology", ["eess.IV", "physics.med-ph"])
]

# ---- Evaluation Metrics ----
def evaluate():
    results = defaultdict(list)
    
    for query, expected_cats in queries:
        for k in K_VALUES:
            try:
                # API Call
                start = time.time()
                response = requests.get(API_URL, params={"query": query, "k": k})
                latency = time.time() - start
                
                if response.status_code != 200:
                    raise ValueError(f"API returned {response.status_code}")
                
                api_results = response.json()
                if not isinstance(api_results, list):
                    raise ValueError("Unexpected API response format")

                # Calculate Metrics
                retrieved = []
                for result in api_results:
                    #paper_cats = metadata[metadata['id'] == result['id']]['categories'].iloc[0]
                    #common_cats = set(paper_cats).intersection(expected_cats)
                    #retrieved.append(len(common_cats) > 0)
                    paper_cats = metadata[metadata['id'] == result['id']]['categories'].iloc[0]
                    retrieved.append(any(c in expected_cats for c in paper_cats))

                #precision = sum(retrieved)/len(retrieved) if retrieved else 0
                #recall = sum(retrieved)/len(expected_cats)
                precision = sum(retrieved)/k
                recall = sum(retrieved)/min(len(expected_cats), k)
                
                # Store Results
                results['query'].append(query)
                results['k'].append(k)
                results['precision'].append(precision)
                results['recall'].append(recall)
                results['latency'].append(latency)
                results['retrieved_ids'].append([r['id'] for r in api_results])
                results['expected_cats'].append(expected_cats)

            except Exception as e:
                print(f"Error evaluating '{query}' @ k={k}: {str(e)}")
                continue

    return pd.DataFrame(results)

def plot_metrics(df):
    # Precision-Recall Curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for k in K_VALUES:
        subset = df[df['k'] == k]
        plt.plot(subset['recall'], subset['precision'], 'o-', label=f'k={k}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # Latency Analysis
    plt.subplot(1, 2, 2)
    df.groupby('k')['latency'].mean().plot(kind='bar')
    plt.ylabel('Latency (s)')
    plt.title('Average Latency by K')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/metrics_plot.png")
    plt.show()

if __name__ == "__main__":
    # Run Evaluation
    eval_df = evaluate()
    
    # Save Results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    eval_df.to_csv(f"{RESULTS_DIR}/eval_results_{timestamp}.csv", index=False)
    
    # Display Summary
    print("\n📊 Evaluation Summary:")
    print(eval_df.groupby('k').agg({
        'precision': 'mean',
        'recall': 'mean',
        'latency': 'mean'
    }).round(3))
    
    # Generate Plots
    plot_metrics(eval_df)