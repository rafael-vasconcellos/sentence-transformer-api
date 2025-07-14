import torch
from sentence_transformers import SentenceTransformer

def download_model():
    model_name = 'all-mpnet-base-v2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Downloading model '{model_name}' to cache...")
    SentenceTransformer(model_name, device=device, cache_folder="./s_cache")
    print("Download complete.")

if __name__ == "__main__":
    download_model()
