
from sentence_transformers import SentenceTransformer
import torch

class TextEmbedder:
    """
    Chuyên tạo embedding cho văn bản dài (README, Docs)
    Sử dụng model: all-MiniLM-L6-v2 (384 dimensions)
    """
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading Text Model ({model_name}) on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("[INFO] Text Model Loaded.")

    def embed(self, text):
        """
        Input: Chuỗi văn bản (str)
        Output: List[float] (384 dimensions)
        """
        if not text or not isinstance(text, str):
            return None
        
        # SentenceTransformer tự lo phần tokenization và encoding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

if __name__ == "__main__":
    # Test nhanh
    embedder = TextEmbedder()
    vec = embedder.embed("This is a React project for e-commerce.")
    print(f"Dimension: {len(vec)}")
    print(f"Sample: {vec[:5]}")
