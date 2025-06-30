############
# embed_papers_ray.py

import ray
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os

# Properly shutdown existing Ray instance if any
if ray.is_initialized():
    ray.shutdown()
ray.init(num_cpus=4, num_gpus=min(1, torch.cuda.device_count()), ignore_reinit_error=True)

@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class EmbedWorker:
    def __init__(self):
        # Load model in FP16 for faster inference
        self.model = SentenceTransformer("allenai/specter", device="cuda")
        self.model.half()  # FP16

    def embed_batch(self, batch):
        try:
            # Combine title + abstract + categories (enhanced input)
            texts = []
            for _, row in batch.iterrows():
                categories = " ".join(row['categories']) if isinstance(row['categories'], list) else str(row['categories'])
                text = f"{row['title']} [SEP] {categories} [SEP] {row['abstract']}"
                texts.append(text)
            
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Return all required fields
            return {
                "ids": batch["id"].tolist(),
                "titles": batch["title"].tolist(),
                "categories": batch["categories"].tolist(),
                "embeddings": embeddings.cpu().numpy().tolist()
            }
        except Exception as e:
            print(f"Failed batch: {str(e)}")
            return None

def run_distributed_embedding(input_path, output_path, batch_size=64, max_workers=2):
    """Run embedding with Ray, optimized for GPU resource constraints."""
    df = pd.read_parquet(input_path)
    print(f"[INFO] Embedding {len(df)} papers (batch_size={batch_size})")

    # Checkpointing: Skip already processed papers
    if os.path.exists(output_path):
        existing = pd.read_parquet(output_path)
        df = df[~df["id"].isin(existing["id"])]
        print(f"[INFO] Resuming from checkpoint. {len(df)} new papers to process.")

    # Split into batches
    batches = [
        df.iloc[i:i + batch_size]
        for i in range(0, len(df), batch_size)
    ]

    # Limit workers to avoid GPU contention
    num_workers = min(max_workers, torch.cuda.device_count())
    workers = [EmbedWorker.remote() for _ in range(num_workers)]

    # Distribute batches to workers
    futures = []
    for batch in batches:
        worker = workers[len(futures) % num_workers]
        futures.append(worker.embed_batch.remote(batch))

    # Collect results with progress bar
    results = []
    for future in tqdm(
        futures,
        desc="Embedding",
        unit="batch"
    ):
        batch_result = ray.get(future)
        if batch_result:
            for paper_id, title, categories, embedding in zip(
                batch_result["ids"],
                batch_result["titles"],
                batch_result["categories"],
                batch_result["embeddings"]
            ):
                results.append({
                    "id": paper_id,
                    "title": title,
                    "categories": categories,
                    "embedding": embedding
                })

    # Merge with existing results (if checkpointing)
    if os.path.exists(output_path):
        existing = pd.read_parquet(output_path)
        results_df = pd.concat([existing, pd.DataFrame(results)])
    else:
        results_df = pd.DataFrame(results)

    # Ensure all required columns are present
    required_columns = ['id', 'title', 'categories', 'embedding']
    for col in required_columns:
        if col not in results_df.columns:
            raise ValueError(f"Missing required column: {col}")

    results_df.to_parquet(output_path, index=False)
    print(f"[DONE] Saved embeddings to {output_path}")

if __name__ == "__main__":
    run_distributed_embedding(
        input_path="papers.parquet",
        output_path="paper_embeddings_ray.parquet",
        batch_size=64,
        max_workers=2
    )