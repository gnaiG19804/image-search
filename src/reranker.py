from sentence_transformers import CrossEncoder

class LocalReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize Local Reranker using a Cross-Encoder model.
        This model runs locally and is much faster than calling an LLM API.
        """
        print(f"[INFO] Loading Reranker Model ({model_name})...")
        self.model = CrossEncoder(model_name)
        print("[INFO] Reranker Model Loaded.")

    def rerank(self, query, candidates, top_k=5):
        """
        Rerank candidates based on relevance to the query.
        
        Args:
            query (str): The search query
            candidates (list): List of project dicts from DB search
            top_k (int): Number of top results to return
            
        Returns:
            list: Top-k reranked candidates
        """
        if not candidates:
            return []

        # Prepare pairs for Cross-Encoder: [ [query, doc_text], ...]
        # We combine title, tags, and domain for a rich representation
        pairs = []
        for c in candidates:
            # Construct a descriptive string for the document
            # E.g. "Amazon Retail Clone. Domain: ecommerce. Tags: ecommerce, retail"
            
            tags_str = ", ".join(c.get('tags', []))
            doc_text = f"{c['title']}. Domain: {c['domain']}. Tags: {tags_str}"
            
            # If backend/frontend available, add them too for tech context
            if c.get('frontend'):
                doc_text += f". Frontend: {', '.join(c['frontend'])}"
            
            pairs.append([query, doc_text])
        
        # Predict scores
        raw_scores = self.model.predict(pairs)
        
        # Apply Sigmoid to normalize to 0-1
        import numpy as np
        scores = 1 / (1 + np.exp(-raw_scores))
        
        # Attach scores to candidates
        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = float(score)
            
        # Sort by new score descending
        ranked_candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        return ranked_candidates[:top_k]
