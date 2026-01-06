
import argparse
import sys
import os
sys.path.append(os.getcwd())
from pathlib import Path
from src.embedding import ImageEmbedder
from src.postgres_db import PostgresDB

def test_search(image_path, limit=5, tech_filter=None):
    print("=" * 60)
    print(f"TEST SEARCH: {image_path}")
    if tech_filter:
        print(f" Filter: Tech Stack contains '{tech_filter}'")
    print("=" * 60)
    
    # 1. Init
    try:
        embedder = ImageEmbedder()
        db = PostgresDB()
    except Exception as e:
        print(f" Init failed: {e}")
        return

    # 2. Embed input image
    input_path = Path(image_path)
    if not input_path.exists():
        print(f" File not found: {image_path}")
        return
        
    try:
        print(" Generating embedding...")
        query_emb = embedder.embed_image(str(input_path))[0].tolist()
    except Exception as e:
        print(f"Embedding failed: {e}")
        return
        
    # 3. Search in DB
    print(" Searching in PostgreSQL...")
    results = db.search_similar(query_emb, limit=limit, filter_tech=tech_filter)
    
    # 4. Display results
    print(f"\n Found {len(results)} matches:\n")
    
    # Header
    print(f"{'#':<3} | {'DISTANCE':<10} | {'PROJECT':<20} | {'IMAGE':<15} | {'TECH/REPO'}")
    print("-" * 80)
    
    for i, row in enumerate(results):
        # row: (title, repo_url, image_name, image_path, distance)
        title, repo_url, img_name, img_path, distance = row
        
        # Distance càng nhỏ càng giống (0 = giống hệt)
        # Similarity = 1 - distance (nếu dùng cosine distance)
        similarity = (1 - distance) * 100
        
        print(f"{i+1:<3} | {distance:.4f}     | {title[:20]:<20} | {img_name[:15]:<15} | {repo_url}")
        print(f"    | Sim: {similarity:.1f}%  | Path: {img_path}")
        print("-" * 80)

    db.close()

if __name__ == "__main__":
    default_img = "test/test_amazon.png"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default=default_img, help="Path to query image")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--tech", help="Filter by Tech Stack (e.g., 'React')")
    
    args = parser.parse_args()
    
    test_search(args.img, args.limit, args.tech)
