"""
Migration Script for Component-Based Image Search System
=========================================================
This script imports data from metadata.json files into the new PostgreSQL schema
with projects, project_images, and components tables.

Usage:
    python migrate_to_postgres.py              # Full migration (drop + recreate)
    python migrate_to_postgres.py --no-drop    # Import without dropping tables
"""

import os
import sys
import json
from pathlib import Path

# Add project root
sys.path.append(os.getcwd())

from src.postgres_db import PostgresDB
from src.embedding import ImageEmbedder


def main():
    """Main migration function"""
    print("=" * 70)
    print(" Component-Based Image Search - Database Migration")
    print("=" * 70)
    
    # Check for --no-drop flag
    drop_tables = "--no-drop" not in sys.argv
    
    # 1. Initialize Database
    db = PostgresDB()
    
    # 2. Apply Schema
    print("\n[Step 1] Applying Schema...")
    if drop_tables:
        print("  -> Dropping existing tables...")
        with db.conn.cursor() as cur:
            cur.execute("""
                DROP TABLE IF EXISTS components CASCADE;
                DROP TABLE IF EXISTS project_images CASCADE;
                DROP TABLE IF EXISTS projects CASCADE;
                DROP TABLE IF EXISTS project_component_embeddings CASCADE;
                DROP TABLE IF EXISTS project_embeddings CASCADE;
                DROP TABLE IF EXISTS project_metadata CASCADE;
                DROP TABLE IF EXISTS project_assets CASCADE;
            """)
        db.conn.commit()
    
    print("  -> Creating new tables from schema.sql...")
    db.init_schema_from_file("schema.sql")
    
    # 3. Load Image Embedder
    print("\n[Step 2] Loading AI Models...")
    print("  -> Loading CLIP Image Embedder (ViT-B/32)...")
    image_embedder = ImageEmbedder()
    print("  -> Models loaded!")
    
    # 4. Scan Dataset
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        print("\n[ERROR] Dataset directory not found!")
        return
    
    projects = sorted(list(dataset_dir.glob("project_*")))
    print(f"\n[Step 3] Found {len(projects)} projects to migrate")
    
    # 5. Migrate Each Project
    print("\n[Step 4] Migrating Projects...\n")
    print("-" * 70)
    
    total_projects = 0
    total_images = 0
    total_components = 0
    
    for idx, project_dir in enumerate(projects, 1):
        meta_path = project_dir / "metadata.json"
        images_dir = project_dir / "images"
        
        if not meta_path.exists():
            print(f"  [{idx}] SKIP: {project_dir.name} - No metadata.json")
            continue
        
        # Load metadata
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        project_code = metadata.get("project_id")
        title = metadata.get("title", "Unknown")
        repo_url = metadata.get("repo_url", "")
        domain = metadata.get("domain", "")
        tech_stack = metadata.get("tech_stack", {})
        
        print(f"\n  [{idx}/{len(projects)}] {title}")
        print(f"       Code: {project_code}")
        print(f"       Repo: {repo_url}")
        
        try:
            # Insert Project
            with db.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO projects (project_code, title, repo_url, domain, frontend, backend, database)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (project_code) DO UPDATE SET
                        title = EXCLUDED.title,
                        repo_url = EXCLUDED.repo_url,
                        domain = EXCLUDED.domain,
                        frontend = EXCLUDED.frontend,
                        backend = EXCLUDED.backend,
                        database = EXCLUDED.database,
                        updated_at = now()
                    RETURNING id
                """, (
                    project_code,
                    title,
                    repo_url,
                    domain,
                    tech_stack.get("frontend", []),
                    tech_stack.get("backend", []),
                    tech_stack.get("database", [])
                ))
                project_uuid = cur.fetchone()[0]
            
            db.conn.commit()
            total_projects += 1
            
            # Process Images
            images_data = metadata.get("images", [])
            project_images_count = 0
            project_components_count = 0
            
            for img_data in images_data:
                image_id = img_data.get("image_id")
                image_path = img_data.get("image_path")
                page_name = img_data.get("page_name")
                components = img_data.get("components", [])
                
                # Full image path
                full_image_path = project_dir / image_path
                
                # Generate image embedding
                image_embedding = None
                if full_image_path.exists():
                    try:
                        embedding_result = image_embedder.embed_image(str(full_image_path))
                        # embed_image returns numpy array with shape (1, 512), flatten it
                        image_embedding = embedding_result.flatten().tolist()
                    except Exception as e:
                        print(f"       [WARN] Could not embed image: {e}")
                
                # Insert Image
                with db.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO project_images (project_id, image_id, image_path, page_name, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        ON CONFLICT (image_id) DO UPDATE SET
                            page_name = EXCLUDED.page_name,
                            embedding = EXCLUDED.embedding
                        RETURNING id
                    """, (
                        project_uuid,
                        image_id,
                        str(project_dir / image_path),
                        page_name,
                        image_embedding
                    ))
                    db_image_id = cur.fetchone()[0]
                
                db.conn.commit()
                project_images_count += 1
                
                # Process Components
                for comp in components:
                    component_id = comp.get("component_id")
                    component_type = comp.get("type")
                    component_name = comp.get("name")
                    semantic_tags = comp.get("semantic_tags", [])
                    description = comp.get("description", "")
                    source_code = comp.get("source_code", {})
                    
                    # Insert Component
                    with db.conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO components (
                                image_id, project_id, component_id, component_type, component_name,
                                semantic_tags, description,
                                source_file_path, source_start_line, source_end_line
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (component_id) DO UPDATE SET
                                component_type = EXCLUDED.component_type,
                                component_name = EXCLUDED.component_name,
                                semantic_tags = EXCLUDED.semantic_tags,
                                description = EXCLUDED.description,
                                source_file_path = EXCLUDED.source_file_path,
                                source_start_line = EXCLUDED.source_start_line,
                                source_end_line = EXCLUDED.source_end_line
                            RETURNING id
                        """, (
                            db_image_id,
                            project_uuid,
                            component_id,
                            component_type,
                            component_name,
                            semantic_tags,
                            description,
                            source_code.get("file_path"),
                            source_code.get("start_line"),
                            source_code.get("end_line")
                        ))
                    
                    project_components_count += 1
                
                db.conn.commit()
            
            total_images += project_images_count
            total_components += project_components_count
            
            print(f"       ✓ {project_images_count} images, {project_components_count} components")
            
        except Exception as e:
            print(f"       ✗ Error: {e}")
            db.conn.rollback()
            continue
    
    # 6. Summary
    print("\n" + "-" * 70)
    print("\n" + "=" * 70)
    print(" Migration Summary")
    print("=" * 70)
    print(f"  ✓ Projects:   {total_projects}")
    print(f"  ✓ Images:     {total_images}")
    print(f"  ✓ Components: {total_components}")
    print("\n[DONE] Migration Complete!\n")
    
    db.close()


def generate_component_embeddings():
    """
    Generate CLIP embeddings for all components by cropping from images.
    Run this after initial migration.
    """
    print("=" * 70)
    print(" Generate Component Embeddings")
    print("=" * 70)
    
    from src.component_embedder import ComponentEmbedder
    from PIL import Image
    
    db = PostgresDB()
    embedder = ComponentEmbedder()
    
    # Get components without embeddings
    with db.conn.cursor() as cur:
        cur.execute("""
            SELECT c.id, c.component_id, c.bbox, pi.image_path
            FROM components c
            JOIN project_images pi ON c.image_id = pi.id
            WHERE c.embedding IS NULL
            ORDER BY c.id
        """)
        components = cur.fetchall()
    
    print(f"\nFound {len(components)} components without embeddings\n")
    
    success = 0
    for idx, (comp_id, component_id, bbox, image_path) in enumerate(components, 1):
        print(f"  [{idx}/{len(components)}] {component_id}...", end=" ")
        
        if not Path(image_path).exists():
            print("Image not found")
            continue
        
        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            
            # If we have bbox, crop the component
            if bbox:
                x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('w', img.width), bbox.get('h', img.height)
                cropped = img.crop((x, y, x + w, y + h))
            else:
                cropped = img
            
            # Generate embedding
            embedding = embedder.embed_image(cropped)
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
            # Update database
            with db.conn.cursor() as cur:
                cur.execute("""
                    UPDATE components SET embedding = %s::vector WHERE id = %s
                """, (embedding_list, comp_id))
            
            db.conn.commit()
            print("✓")
            success += 1
            
        except Exception as e:
            print(f"Error: {str(e)[:40]}")
            db.conn.rollback()
    
    print(f"\n[DONE] Generated {success}/{len(components)} embeddings\n")
    db.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "embeddings":
        generate_component_embeddings()
    else:
        main()
