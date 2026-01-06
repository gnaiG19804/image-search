

import faiss
import json
from pathlib import Path
from collections import defaultdict


class ProjectSearcher:
    """
    Engine tìm kiếm dự án tương tự dựa trên hình ảnh
    
    """
    
    def __init__(self, embedder, index_path="index/index.faiss", meta_path="index/metadata.json"):
        """
        Khởi tạo ProjectSearcher
        

        """
        self.embedder = embedder
        
        # Load FAISS index
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Không tìm thấy index: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        print(f"Đã load index: {self.index.ntotal} vectors")
        
        # Load metadata
        meta_path = Path(meta_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Không tìm thấy metadata: {meta_path}")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"Đã load metadata: {len(self.metadata)} items")
    
    def search(self, query_image_path, top_k=5, search_k=None):
        """
        Tìm các dự án tương tự với ảnh query

        """
        # Tạo embedding cho ảnh query
        print(f"Đang tìm kiếm cho: {query_image_path}")
        query_emb = self.embedder.embed_image(query_image_path)
        
        # Tìm kiếm K-nearest neighbors
        # Lấy nhiều kết quả hơn top_k vì mỗi project có nhiều ảnh
        if search_k is None:
            search_k = top_k * 3
        
        search_k = min(search_k, self.index.ntotal)  # Không vượt quá tổng số vectors
        
        scores, indices = self.index.search(query_emb, search_k)
        
        # Group kết quả theo project_id
        project_matches = defaultdict(list)
        
        for score, idx in zip(scores[0], indices[0]):
            # Bỏ qua nếu index không hợp lệ
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            project_id = meta['project_id']
            
            # Lưu thông tin match
            project_matches[project_id].append({
                'score': float(score),
                'image_name': meta['image_name'],
                'image_path': meta['image_path'],
                'metadata': meta
            })
        
        # Chọn match tốt nhất của mỗi project
        results = []
        
        for project_id, matches in project_matches.items():
            # Lấy match có điểm cao nhất
            best_match = max(matches, key=lambda x: x['score'])
            meta = best_match['metadata']
            
            # Tạo result object
            result = {
                'project_id': project_id,
                'title': meta.get('title', 'N/A'),
                'similarity_score': best_match['score'],
                'matched_image': best_match['image_name'],
                'matched_image_path': best_match['image_path'],
                'tech_stack': meta.get('tech_stack', []),
                'estimate_days': meta.get('estimate_days', 'N/A'),
                'repo_url': meta.get('repo_url', ''),
                'tags': meta.get('tags', []),
                'description': meta.get('description', ''),
                'num_matches': len(matches),  # Số lượng ảnh khớp của project này
            }
            
            results.append(result)
        
        # Sắp xếp theo điểm similarity (cao -> thấp)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Trả về top_k kết quả
        return results[:top_k]
    
    def search_with_filters(self, query_image_path, top_k=5, filters=None):
        """
        Tìm kiếm với filters (lọc theo tech_stack, tags, estimate time, etc.)
        """
        # Tìm kiếm thông thường trước
        # Lấy nhiều kết quả hơn để sau khi filter vẫn đủ top_k
        results = self.search(query_image_path, top_k=top_k * 3)
        
        if filters is None:
            return results[:top_k]
        
        filtered_results = []
        
        for result in results:
            # Kiểm tra filters
            if 'tech_stack' in filters:
                required_tech = filters['tech_stack']
                if not any(tech in result['tech_stack'] for tech in required_tech):
                    continue
            
            if 'tags' in filters:
                required_tags = filters['tags']
                if not any(tag in result['tags'] for tag in required_tags):
                    continue
            
            if 'max_days' in filters:
                if result['estimate_days'] != 'N/A':
                    if result['estimate_days'] > filters['max_days']:
                        continue
            
            if 'min_similarity' in filters:
                if result['similarity_score'] < filters['min_similarity']:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results[:top_k]


def test_searcher():
    print("TEST PROJECT SEARCHER")
    
    from embedding import ImageEmbedder
    
    # Kiểm tra index có tồn tại không
    index_path = Path("index/index.faiss")
    if not index_path.exists():
        print(" Chưa có index!")
        return
    
    # Tạo searcher
    embedder = ImageEmbedder()
    searcher = ProjectSearcher(embedder)
    
    # Test search với ảnh trong dataset
    dataset_path = Path("dataset")
    
    # Tìm ảnh đầu tiên để test
    test_image = "test\\test_amazon.png"
    
    print(f"Test search với: {test_image}")
    
    # Search
    results = searcher.search(test_image, top_k=3)
    
    # In kết quả
    print(f"KẾT QUẢ TÌM KIẾM")
    
    for i, result in enumerate(results, 1):
        print(f"#{i} {result['title']}")
        print(f"   Similarity: {result['similarity_score']:.3f}")
        print(f"   Matched: {result['matched_image']}")
        print(f"   Tech: {', '.join(result['tech_stack'])}")
        print()


if __name__ == "__main__":
    test_searcher()
