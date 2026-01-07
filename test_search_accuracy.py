from src.brain.llm_parser import LLMParser
from src.postgres_db import PostgresDB
from src.text_embedder import TextEmbedder
from src.reranker import LocalReranker

def test_search():
    print("Initializing...")
    llm = LLMParser()
    db = PostgresDB()
    embedder = TextEmbedder()
    reranker = LocalReranker()

    queries = [
        "cho tôi dự án thương mại điện tử sử dụng React",
        "tìm app học ngôn ngữ",
        "fintech app uy tín",
        "dự án giống amazon"
    ]

    for q in queries:
        print(f"\n\nTest Query: '{q}'")
        print("-" * 50)
        
        # 1. Parse
        parsed = llm.parse_query_v2(q)
        print(f"Parsed: {parsed}")
        
        # 2. Embed
        semantic_query = parsed.get('semantic_query', q)
        print(f"Semantic Query: '{semantic_query}'")
        vector = embedder.embed(semantic_query)
        
        # 3. Search Candidates
        print(f"Searching for candidates...")
        candidates = db.search_projects(query_vector=vector, filters=parsed.get('filters'), limit=10)
        
        # 4. Rerank
        print(f"Reranking {len(candidates)} candidates...")
        results = reranker.rerank(q, candidates, top_k=3)
        
        # 5. Show results
        for idx, r in enumerate(results):
            # Try to get score from rerank_score if available, else original score
            score = r.get('rerank_score', r.get('score', 0))
            print(f"[{idx+1}] {r['title']} (Score: {score:.3f})")
            print(f"    Domain: {r['domain']} | Tags: {r['tags'][:3]}")

if __name__ == "__main__":
    test_search()
