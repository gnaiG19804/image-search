
import uuid
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from src.embedding import ImageEmbedder
from src.postgres_db import PostgresDB

def migrate_to_postgres(dataset_path="./dataset"):
    print("=" * 50)
    print("PRODUCTION MIGRATION: DATASET -> POSTGRESQL")
    print("=" * 50)
    
    # 1. Kết nối DB
    try:
        db = PostgresDB() # Dùng default creds trong docker-compose
        print("Connected to DB.")
    except:
        print(" Cannot connect to Postgres. Make sure Docker is running!")
        return

    # 2. Init Embedder
    try:
        embedder = ImageEmbedder()
        print("Model Loaded.")
    except Exception as e:
        print(f" Error loading model: {e}")
        return
        
    # 3. Scan & Import
    dataset_dir = Path(dataset_path)
    total_images = 0
    
    print("\nStarting migration...")
    
    for project_dir in tqdm(list(dataset_dir.iterdir()), desc="Projects"):
        if not project_dir.is_dir(): continue
        
        # Metadata
        meta_file = project_dir / "metadata.json"
        if not meta_file.exists(): continue
        
        with open(meta_file, encoding='utf-8') as f:
            meta = json.load(f)
            
        if 'project_id' not in meta:
            meta['project_id'] = project_dir.name
            
        # A. Add Project
        db.add_project(meta)
        
        # B. Add Images & Vectors
        image_files = list(project_dir.rglob("*.png")) + list(project_dir.rglob("*.jpg"))
        
        for img_file in image_files:
            try:
                # Embed
                emb_res = embedder.embed_image(str(img_file))
                vector = emb_res[0].tolist()
                
                # Generate ID
                image_id = str(uuid.uuid4())
                
                # Add to DB
                db.add_image_with_embedding(
                    image_id=image_id,
                    project_id=meta['project_id'],
                    image_path=str(img_file),
                    image_name=img_file.name,
                    vector=vector
                )
                total_images += 1
            except Exception as e:
                print(f"  Error {img_file.name}: {e}")
                
    print("\n" + "=" * 50)
    print(f"COMPLETE! Imported {total_images} images.")
    print(f"Total in DB: {db.count_images()}")
    db.close()

if __name__ == "__main__":
    migrate_to_postgres()
