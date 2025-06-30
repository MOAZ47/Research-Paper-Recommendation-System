#########
# 6. api/main.py
#########

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- Load Models & Index ----------
model = SentenceTransformer("allenai/specter")
index = faiss.read_index("faiss_index/papers.index")
metadata = pd.read_parquet("faiss_index/metadata.parquet")

# ---------- API Setup ----------
app = FastAPI(title="Research Paper Recommender")

class SearchResult(BaseModel):
    id: str
    title: str
    score: float
    categories: List[str]

@app.get("/search", response_model=List[SearchResult])
def search(query: str = Query(...), k: int = 5):
    # Embed and normalize
    vec = model.encode([query])
    faiss.normalize_L2(vec)

    scores, indices = index.search(vec, k)
    results = metadata.iloc[indices[0]].copy()
    results["score"] = scores[0]

    return [
        SearchResult(id=row["id"], title=row["title"], score=row["score"], categories=row['categories'])
        for _, row in results.iterrows()
    ]

# uvicorn api.main:app --reload