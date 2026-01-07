
import os
import sys
import json
from pathlib import Path

# Add project root
sys.path.append(os.getcwd())

from src.postgres_db import PostgresDB
from src.text_embedder import TextEmbedder
from src.embedding import ImageEmbedder

def main():
    print("="*60)
    print(" Migration to PostgreSQL Schema V2 (Normalized)")
    print("="*60)
    
    # 1. Initialize Database (will create connection)
    db = PostgresDB()
    
    # 2. Apply Schema from schema.sql
    print("\n [Step 1] Applying Schema...")
    db.init_schema_from_file("schema.sql")
    
    # 3. Initialize Embedders
    print("\n [Step 2] Loading AI Models...")
    print("  → Loading Text Embedder (paraphrase-multilingual-MiniLM-L12-v2)...")
    text_embedder = TextEmbedder()
    
    print("  → Loading Image Embedder (CLIP ViT-B/32)...")
    image_embedder = ImageEmbedder()
    
    # 4. Scan Dataset
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        print(" [ERROR] Dataset directory not found!")
        return
    
    projects = sorted(list(dataset_dir.glob("project_*")))
    print(f"\n [Step 3] Found {len(projects)} projects to migrate")
    
    # 5. Migrate Each Project
    print("\n [Step 4] Migrating Projects...\n")
    
    for idx, project_dir in enumerate(projects, 1):
        meta_path = project_dir / "metadata.json"
        images_dir = project_dir / "images"
        
        if not meta_path.exists():
            print(f"  [WARN] [{idx}/{len(projects)}] Skipping {project_dir.name} - No metadata.json")
            continue
        
        # Load metadata
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"  [{idx}/{len(projects)}] Processing: {metadata.get('title', 'Unknown')}")
        
        try:
            # Insert project with all related data
            proj_uuid = db.add_project(
                metadata=metadata,
                text_embedder=text_embedder,
                image_embedder=image_embedder,
                images_dir=images_dir if images_dir.exists() else None
            )
            
            # Count images processed
            if images_dir.exists():
                img_count = len(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
                print(f"      [OK] Inserted project with {img_count} images")
            else:
                print(f"      [OK] Inserted project (no images)")
                
        except Exception as e:
            print(f"      [ERROR]    Error: {e}")
            continue
    
    # 6. Summary
    print("\n" + "="*60)
    print(" Migration Summary:")
    print("="*60)
    
    project_count = db.count_projects()
    image_count = db.count_images()
    
    print(f"[OK] Total Projects: {project_count}")
    print(f"[OK] Total Images: {image_count}")
    print("\n[DONE] Migration Complete! Database is ready for production.\n")
    
    db.close()

if __name__ == "__main__":
    main()
