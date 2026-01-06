
import faiss
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


class ProjectIndexer:

    
    def __init__(self, embedder):
        """
        Khởi tạo ProjectIndexer
        
        """
        self.embedder = embedder
        self.index = None           # FAISS index
        self.metadata = []          # Metadata của các project/image
        self.dimension = 512        # Dimension của CLIP ViT-B/32
    
    def build_index(self, dataset_path, index_type="flat"):
        """
        Scan dataset và tạo FAISS index
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_path}")
        
        print(f"\n Đang quét dataset tại: {dataset_path}")
        
        embeddings = []
        metadata = []
        
        # Lấy danh sách các thư mục project
        project_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        if not project_dirs:
            raise ValueError(f"Không tìm thấy thư mục project nào trong {dataset_path}")
        
        print(f" Tìm thấy {len(project_dirs)} thư mục project\n")
        
        # Duyệt qua từng project
        for project_dir in tqdm(project_dirs, desc="Xử lý projects"):
            # Đọc metadata
            meta_file = project_dir / "metadata.json"
            
            if not meta_file.exists():
                print(f"  Bỏ qua {project_dir.name}: không có metadata.json")
                continue
            
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    project_meta = json.load(f)
            except Exception as e:
                print(f"  Lỗi đọc metadata {project_dir.name}: {e}")
                continue
            
            # Tìm tất cả ảnh trong project (bao gồm subfolder)
            image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
            image_files_set = set()  # Dùng set để tránh duplicate
            
            for ext in image_extensions:
                # Tìm recursive trong tất cả subfolder
                image_files_set.update(project_dir.rglob(f'*{ext}'))
                image_files_set.update(project_dir.rglob(f'*{ext.upper()}'))
            
            image_files = list(image_files_set)  # Convert set về list
            
            if not image_files:
                print(f"  Không tìm thấy ảnh trong {project_dir.name}")
                continue
            
            # Tạo embedding cho từng ảnh
            for img_file in image_files:
                try:
                    # Tạo embedding
                    emb = self.embedder.embed_image(img_file)
                    embeddings.append(emb[0])
                    
                    # Lưu metadata kèm info về ảnh
                    metadata.append({
                        **project_meta,
                        'image_path': str(img_file),
                        'image_name': img_file.name,
                        'project_folder': project_dir.name
                    })
                    
                except Exception as e:
                    print(f"  Lỗi xử lý {img_file.name}: {e}")
                    continue
        
        if not embeddings:
            raise ValueError("Không tạo được embedding nào! Kiểm tra lại dataset.")
        
        # Chuyển embeddings thành numpy array
        embeddings = np.array(embeddings).astype('float32')
        
        print(f"\n Đã tạo {len(embeddings)} embeddings")
        
        # Tạo FAISS index
        self._create_faiss_index(embeddings, index_type)
        
        # Lưu metadata
        self.metadata = metadata
        
        num_projects = len(set(m['project_id'] for m in metadata))
        
        return len(embeddings), num_projects
    
    def _create_faiss_index(self, embeddings, index_type):
        """
        Tạo FAISS index từ embeddings
        
        """
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        print(f"\nĐang tạo FAISS index ({index_type})...")
        
        if index_type == "flat":
            # IndexFlatIP: Inner Product (= Cosine similarity cho normalized vectors)
            # Chính xác 100%, phù hợp cho < 1M vectors
            self.index = faiss.IndexFlatIP(dimension)
            
        elif index_type == "ivf":
            # IVF: Inverted File Index (faster, approximate)
            quantizer = faiss.IndexFlatIP(dimension)
            nlist = min(100, n_vectors // 10)  # Số clusters
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(embeddings)
            
        elif index_type == "hnsw":
            # HNSW: Hierarchical Navigable Small World (very fast)
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
            
        else:
            raise ValueError(f"Index type không hợp lệ: {index_type}")
        
        # Add vectors vào index
        self.index.add(embeddings)
        
        print(f" Index created: {n_vectors} vectors, dimension={dimension}")
    
    def save(self, index_path="index.faiss", meta_path="metadata.json"):
        """
        Lưu index và metadata ra file
        
        """
        if self.index is None:
            raise ValueError("Index chưa được tạo! Gọi build_index() trước.")
        
        # Lưu FAISS index
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        print(f" Đã lưu index: {index_path}")
        
        # Lưu metadata
        meta_path = Path(meta_path)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f" Đã lưu metadata: {meta_path}")
    
    def load(self, index_path="index.faiss", meta_path="metadata.json"):
        """
        Load index và metadata từ file
        """
        # Load FAISS index
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Không tìm thấy index: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        print(f" Đã load index: {self.index.ntotal} vectors")
        
        # Load metadata
        meta_path = Path(meta_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Không tìm thấy metadata: {meta_path}")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f" Đã load metadata: {len(self.metadata)} items")


def test_indexer():
    """Test function"""
    from embedding import ImageEmbedder
    
    # Kiểm tra dataset có tồn tại không
    dataset_path = Path("dataset")
    if not dataset_path.exists() or not list(dataset_path.iterdir()):
        print(" Chưa có dataset để test!")
        return
    
    # Tạo embedder
    embedder = ImageEmbedder()
    
    # Tạo indexer
    indexer = ProjectIndexer(embedder)
    
    # Build index
    try:
        n_images, n_projects = indexer.build_index(dataset_path)
        print(f"\n Build index thành công!")
        print(f"   - {n_images} images")
        print(f"   - {n_projects} projects")
        
        # Save index
        indexer.save("index/index.faiss", "index/metadata.json")
        
    except Exception as e:
        print(f" Lỗi: {e}")


if __name__ == "__main__":
    test_indexer()
