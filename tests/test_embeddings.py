import pandas as pd

def test_embeddings_file():
    """Test if embeddings file exists and is valid"""
    try:
        df = pd.read_parquet("paper_embeddings_ray.parquet")
        assert not df.empty
        assert "embedding" in df.columns
        print("✅ Embeddings File Test Passed")
    except Exception as e:
        print(f"❌ Embeddings Test Failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_embeddings_file()