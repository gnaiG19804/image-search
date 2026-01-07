
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.postgres_db import PostgresDB
from src.llm.llm_parser import LLMParser
from src.text_embedder import TextEmbedder
from src.reranker import LocalReranker

def print_separator():
    print("\n" + "="*70)

def print_result(result, rank):
    """Pretty print a search result"""
    print(f"\n[{rank}] {result['title']}")
    print(f"    ID: {result['project_code']}")
    print(f"    Domain: {result['domain']} | Platform: {', '.join(result['platform'])}")
    
    # Tech stack
    frontend_str = ', '.join(result['frontend']) if result['frontend'] else 'N/A'
    backend_str = ', '.join(result['backend']) if result['backend'] else 'N/A'
    print(f"    Frontend: {frontend_str}")
    print(f"    Backend:  {backend_str}")
    
    # Metadata
    print(f"    Complexity: {result['complexity']} | Team Size: {result['team_size']}")
    print(f"    Tags: {', '.join(result['tags'][:5]) if result['tags'] else 'None'}")
    print(f"    Score: {result['score']:.3f}")
    
    if result.get('repo_url'):
        print(f"    Repo: {result['repo_url']}")

def search_interactive():
    """
    Interactive Search CLI with Schema V2 Support
    """
    print_separator()
    print("Interactive Project Search (Schema V2)")
    print("Features: Metadata Filtering + Vector Semantic Search")
    print_separator()
    
    # Initialize components
    print("\nLoading AI Models...")
    llm = LLMParser()
    text_embedder = TextEmbedder()
    reranker = LocalReranker()
    db = PostgresDB()
    
    print("Ready!\n")
    
    # Stats
    total_projects = db.count_projects()
    total_images = db.count_images() 
    print(f"Database: {total_projects} projects, {total_images} images\n")
    
    while True:
        print_separator()
        user_query = input("\nEnter search query (or 'exit' to quit): ").strip()
        
        if not user_query or user_query.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
        
        print("\nAnalyzing query...")
        
        # Step 1: Parse query with LLM V2
        parsed = llm.parse_query_v2(user_query)
        
        print(f"\nExtracted Filters:")
        print(f"   Intent: {parsed.get('intent', 'N/A')}")
        
        filters = parsed.get('filters', {})
        if filters:
            if filters.get('domain'):
                print(f"   Domain: {filters['domain']}")
            if filters.get('frontend'):
                print(f"   Frontend: {filters['frontend']}")
            if filters.get('backend'):
                print(f"   Backend: {filters['backend']}")
            if filters.get('platform'):
                print(f"   Platform: {filters['platform']}")
            if filters.get('complexity'):
                print(f"   Complexity: {filters['complexity']}")
            if filters.get('tags'):
                print(f"   Tags: {filters['tags']}")
        else:
            print("   (No specific filters)")
        
        semantic_query = parsed.get('semantic_query', user_query)
        print(f"   Semantic Query: \"{semantic_query}\"")
        
        # Step 2: Generate embedding for semantic search
        print("\nGenerating embedding...")
        query_vector = text_embedder.embed(semantic_query)
        
        # Step 3: Execute hybrid search
        print(f"Searching database for candidates...")
        
        candidates = db.search_projects(
            query_vector=query_vector,
            filters=filters,
            limit=20  # Fetch more candidates for reranking
        )
        
        if not candidates:
            print("No projects match your criteria.")
            # ... tips ...
            continue

        print(f"Found {len(candidates)} candidates. Reranking (Local AI)...")
        
        # Step 4: Local Reranking (Use semantic_query for English model)
        results = reranker.rerank(semantic_query, candidates, top_k=5)
        
        # Step 5: Display results (Filter > 60%)
        results = [r for r in results if r.get('rerank_score', 0) > 0.6]
        
        print_separator()
        if not results:
             print("\nNo high-confidence matches found (>60%). Try adjusting your query.\n")
        else:
             print(f"\nTop {len(results)} Matches (>60% relevant):\n")
        
        for idx, result in enumerate(results, 1):
            # Update score for display to match valid ranking order
            if 'rerank_score' in result:
                result['score'] = result['rerank_score']
            print_result(result, idx)
        
        print_separator()

def main():
    if len(sys.argv) > 1:
        # Non-interactive mode (single query)
        query = " ".join(sys.argv[1:])
        
        llm = LLMParser()
        text_embedder = TextEmbedder()
        db = PostgresDB()
        
        parsed = llm.parse_query_v2(query)
        vector = text_embedder.embed(parsed.get('semantic_query', query))
        results = db.search_projects(query_vector=vector, filters=parsed.get('filters', {}), limit=5)
        
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} results:\n")
        for idx, r in enumerate(results, 1):
            print(f"{idx}. {r['title']} ({r['domain']}) - Score: {r['score']:.3f}")
    else:
        # Interactive mode
        search_interactive()

if __name__ == "__main__":
    main()
