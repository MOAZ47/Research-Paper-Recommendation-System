#########
# 4. scripts/build_faiss_index.py
#########

import faiss
import numpy as np
import pandas as pd
import os

def build_faiss_index(parquet_path, index_path, nlist=100):
    df = pd.read_parquet(parquet_path)
    embeddings = np.stack(df["embedding"].values).astype('float32')
    faiss.normalize_L2(embeddings)  # Required for cosine similarity

    dim = embeddings.shape[1]
    
    # Use IVFPQ for large datasets (faster search)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, 8, 8)  # 8-bit quantization
    index.train(embeddings)
    index.add(embeddings)

    # Save index and metadata
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, f"{index_path}/papers.index")
    
    # Only keep essential metadata
    df[["id", "title", "categories"]].to_parquet(f"{index_path}/metadata.parquet", index=False)
    print(f"[DONE] FAISS index built with {len(df)} papers.")

if __name__ == "__main__":
    build_faiss_index("paper_embeddings_ray.parquet", "faiss_index")