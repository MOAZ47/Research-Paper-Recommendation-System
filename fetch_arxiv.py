#############
# 1. fetch_arxiv.py
#############

import arxiv
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm

def fetch_recent_papers(query="cs.AI", max_results=1000, days_back=7):
    """Fetch papers from arXiv, only those submitted in the last `days_back` days."""
    client = arxiv.Client()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

    search = arxiv.Search(
        query=f"cat:{query}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    papers = []

    def fetch_paper(result):
        try:
            if result and result.title and result.summary and result.published >= cutoff_date:
                return {
                    "id": result.get_short_id(),
                    "title": result.title,
                    "abstract": result.summary,
                    "categories": result.categories,
                    "authors": [a.name for a in result.authors],
                    "published": str(result.published),
                    "url": result.entry_id
                }
        except Exception as e:
            print(f"Error fetching paper: {e}")
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        try:
            for result in client.results(search):
                futures.append(executor.submit(fetch_paper, result))
        except arxiv.UnexpectedEmptyPageError:
            print("Reached the end of results.")
            pass

        for future in futures:
            paper = future.result()
            if paper:
                papers.append(paper)

    print(f"Fetched {len(papers)} papers for query '{query}'")
    return pd.DataFrame(papers)

if __name__ == "__main__":
    list_categories = ['astro-ph', 'cond-mat', 'cs', 'econ', 'eess', 'gr-qc', 'math', 'physics', 'stat']
    
    for category in tqdm(list_categories, desc="Fetching arXiv categories"):
      print(f"\nFetching papers for {category}")
      
      # Example: Fetch recent ML papers (last 7 days)
      df = fetch_recent_papers(category, max_results=100, days_back=7)
      time.sleep(2)  # Rate limiting

      if df.empty:
        print(f"No new papers for {category}. Skipping...")
        continue
      
      if os.path.exists("papers.parquet"):
          existing_df = pd.read_parquet("papers.parquet")
          df = pd.concat([existing_df, df]).drop_duplicates("id")

      df.to_parquet("papers.parquet", index=False)
      print(f"Saved {len(df)} papers to papers.parquet related to {category} \n")