

import torch
import clip
from PIL import Image
from pathlib import Path


class ImageEmbedder:
    """
    Biến hình ảnh thành vector embedding để so sánh tương đồng
    """
    
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        Khởi tạo CLIP model
        """

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load CLIP model và preprocessing function
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f" Đã load model thành công!")
    
    def embed_image(self, image_path):
        """
        Tạo embedding vector từ một hình ảnh
        """
        # Kiểm tra file tồn tại
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {image_path}")
        
        try:
            # Mở và xử lý hình ảnh
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tạo embedding (không tính gradient vì chỉ inference)
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                
                # Normalize vector về độ dài = 1 (để tính cosine similarity)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            # Chuyển về numpy array
            return embedding.cpu().numpy()
            
        except Exception as e:
            raise ValueError(f"Lỗi khi xử lý hình ảnh {image_path}: {str(e)}")
    
    def embed_batch(self, image_paths, batch_size=32):
        """
        Tạo embedding cho nhiều hình ảnh cùng lúc
        """
        embeddings = []
        
        # Chia thành các batch nhỏ
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Load và preprocess các ảnh trong batch
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    tensor = self.preprocess(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"  Bỏ qua {path}: {str(e)}")
                    continue
            
            if not batch_tensors:
                continue
            
            # Stack thành batch tensor
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Tạo embeddings
            with torch.no_grad():
                batch_embeddings = self.model.encode_image(batch_tensor)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate tất cả batches
        if embeddings:
            import numpy as np
            return np.vstack(embeddings)
        else:
            import numpy as np
            return np.array([])


def test_embedder():
    
    # Khởi tạo embedder
    embedder = ImageEmbedder()
    
    # Thông tin về model
    print(f" Device: {embedder.device}")
    print(f" Model: {embedder.model.__class__.__name__}")
    
    # Test với một ảnh (cần có ảnh test)
    test_image = Path("test\\test_amazon.png")
    if test_image.exists():
        print(f"\nTesting với {test_image}...")
        embedding = embedder.embed_image(test_image)
        print(f" Embedding shape: {embedding.shape}")
        print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
        print(f" Vector norm: {(embedding**2).sum()**0.5:.3f}")
    else:
        print(f"\n  Không tìm thấy {test_image} để test")
        print(" Tạo file test.png để test embedder")


if __name__ == "__main__":
    test_embedder()
