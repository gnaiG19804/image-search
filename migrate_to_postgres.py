
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
    print("  -> Loading Text Embedder (paraphrase-multilingual-MiniLM-L12-v2)...")
    text_embedder = TextEmbedder()
    
    print("  -> Loading Image Embedder (CLIP ViT-B/32)...")
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



def update_component_metadata():
    """
    Update component metadata for existing images in database
    Process one project at a time with automatic resume capability
    """
    print("="*60)
    print(" Update Component Metadata (Project-by-Project)")
    print("="*60)
    
    # Import here to avoid loading heavy models if not needed
    from src.component_detector import UIComponentDetector
    from src.component_utils import build_components_metadata, metadata_to_json
    from src.component_embedder import ComponentEmbedder
    
    # Initialize database
    db = PostgresDB()
    
    # Initialize detector
    print("\n [Step 1] Loading Component Detector (SAM + CLIP)...")
    print("-> This may take a while on first run (downloading models)...")
    detector = UIComponentDetector(
        method='sam',
        classify_semantics=True,
        use_clip=True
    )
    print("Detector loaded!")
    
    print("\n [Step 1b] Loading Component Embedder (CLIP)...")
    embedder = ComponentEmbedder()
    print("Embedder loaded!")
    
    # Get projects that need component detection (group by project)
    print("\n [Step 2] Finding projects to process...")
    
    # Check for limit from command line argument
    limit = None
    if len(sys.argv) > 2:
        try:
            limit = int(sys.argv[2])
            print(f"  -> Limiting to first {limit} projects")
        except ValueError:
            pass
    
    with db.conn.cursor() as cur:
        query = """
            SELECT 
                p.id as project_id,
                p.title,
                COUNT(pi.id) as total_images,
                COUNT(CASE WHEN pi.components_metadata IS NULL THEN 1 END) as pending_images
            FROM projects p
            JOIN project_images pi ON p.id = pi.project_id
            GROUP BY p.id, p.title
            HAVING COUNT(CASE WHEN pi.components_metadata IS NULL THEN 1 END) > 0
            ORDER BY p.title
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cur.execute(query)
        projects = cur.fetchall()
    
    if len(projects) == 0:
        print("\nAll projects already have component metadata!")
        db.close()
        return
    
    total_projects = len(projects)
    total_pending_images = sum(p[3] for p in projects)
    
    print(f"  -> Found {total_projects} projects with {total_pending_images} images to process")
    print(f"  -> This allows resume if crashed!\n")
    
    # Process each project
    print(" [Step 3] Processing projects...\n")
    print("-" * 60)
    
    overall_success = 0
    overall_errors = 0
    completed_projects = 0
    
    for proj_idx, (project_id, title, total_imgs, pending_imgs) in enumerate(projects, 1):
        print(f"\nPROJECT [{proj_idx}/{total_projects}]: {title}")
        print(f"   Total images: {total_imgs} | Pending: {pending_imgs}")
        print("-" * 60)
        
        # Get images for this project
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT id, image_path, image_name
                FROM project_images
                WHERE project_id = %s AND components_metadata IS NULL
                ORDER BY id
            """, (project_id,))
            images = cur.fetchall()
        
        project_success = 0
        project_errors = 0
        
        # Process each image in this project
        for img_idx, (img_id, path_str, img_name) in enumerate(images, 1):
            path = Path(path_str)
            
            print(f"   [{img_idx}/{len(images)}] {img_name}...", end=" ")
            
            if not path.exists():
                print(f"File not found")
                project_errors += 1
                continue
            
            try:
                # Detect components
                components = detector.detect(str(path))
                
                # Build metadata
                metadata = build_components_metadata(components)
                json_meta = metadata_to_json(metadata)
                
                # Update database for this image
                with db.conn.cursor() as cur:
                    cur.execute("""
                        UPDATE project_images 
                        SET components_metadata = %s 
                        WHERE id = %s
                    """, (json_meta, img_id))
                
                # 3. Generate Embeddings & Save to project_component_embeddings
                # Generate embeddings for all components
                components = embedder.embed_components(str(path), components)
                
                with db.conn.cursor() as cur:
                    # Clear old embeddings for this image first (to avoid duplicates if re-running)
                    cur.execute("DELETE FROM project_component_embeddings WHERE image_id = %s", (img_id,))
                    
                    # Insert new embeddings
                    for idx, comp in enumerate(components):
                        bbox = comp.get('bbox', [0, 0, 0, 0])
                        # Calculate area ratio if possible, else 0
                        # Assuming image size available, but if not easily accessible here, we skip or approximate
                        # We have img_w, img_h in detector but here we iterate. 
                        # detector.detect returns dicts. embedder.embed_components needs image path to load image anyway.
                        # Let's trust embedder populated embedding.
                        
                        cur.execute("""
                            INSERT INTO project_component_embeddings (
                                project_id, image_id,
                                component_type, component_index,
                                bbox, bbox_norm,
                                embedding, confidence
                            ) VALUES (
                                %s, %s,
                                %s, %s,
                                %s::jsonb, %s::jsonb,
                                %s::vector, %s
                            )
                        """, (
                            project_id, 
                            img_id,
                            comp.get('semantic_type', 'unknown'),
                            idx,
                            json.dumps(comp.get('bbox')),
                            json.dumps(comp.get('bbox_norm')),
                            comp.get('embedding'), # List of 512 floats
                            comp.get('confidence', 1.0)
                        ))
                
                # Commit immediately after each image (safety)
                db.conn.commit()
                
                # Log summary
                type_counts = {k: v['count'] for k, v in metadata.get('by_type', {}).items()}
                print(f"✓ {len(components)} components: {type_counts}")
                
                project_success += 1
                
            except Exception as e:
                print(f"Error: {str(e)[:50]}")
                project_errors += 1
                db.conn.rollback()
        
        # Project completed - show summary
        overall_success += project_success
        overall_errors += project_errors
        completed_projects += 1
        
        print(f"\n   Project complete: {project_success} success, {project_errors} errors")
        print(f"   Changes committed to database")
        print("-" * 60)
    
    # Final Summary
    print("\n" + "="*60)
    print(" FINAL SUMMARY")
    print("="*60)
    print(f"Completed projects: {completed_projects}/{total_projects}")
    print(f"Successful images: {overall_success}")
    print(f"Failed images: {overall_errors}")
    print(f"Total processed: {overall_success + overall_errors}")
    print("\nAll changes saved to database!")
    print("Safe to resume if interrupted")
    print("\n[DONE] Component metadata updated!\n")
    
    db.close()



if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "update-components":
        update_component_metadata()
    else:
        main()

