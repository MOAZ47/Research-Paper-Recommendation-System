#########
# 5. scripts/search.py
#########

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load FAISS index and metadata
index = faiss.read_index("faiss_index/papers.index")
metadata = pd.read_parquet("faiss_index/metadata.parquet")

# Load embedding model
model = SentenceTransformer("allenai/specter")

def embed_query(text):
    vec = model.encode([text])
    faiss.normalize_L2(vec)
    return vec

def search_papers(query, k=5):
    vec = embed_query(query)
    scores, indices = index.search(vec, k)
    results = metadata.iloc[indices[0]].copy()
    results["score"] = scores[0]
    return results

if __name__ == "__main__":
    while True:
        q = input("\n🔍 Enter a paper title or abstract (or 'q' to quit):\n> ")
        if q.lower() == "q":
            break
        results = search_papers(q)
        print("\n📄 Top Results:")
        for i, row in results.iterrows():
            print(f"- {row['title']}  [Score: {row['score']:.3f}]")