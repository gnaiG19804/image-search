
import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
import uuid
import json

class PostgresDB:
    """
    Database Manager for Schema V2 (Normalized 5-Table Structure)
    """
    
    def __init__(self, dsn="postgresql://user:password@localhost:5433/vectordb"):
        self.dsn = dsn
        self.conn = None
        self.connect()
        
    def connect(self):
        try:
            self.conn = psycopg2.connect(self.dsn)
            self.conn.autocommit = True
            register_vector(self.conn)
            print(" Connected to PostgreSQL")
        except Exception as e:
            print(f" Connection Failed: {e}")
            raise e

    def init_schema_from_file(self, schema_path="schema.sql"):
        """Apply schema from SQL file (drops old tables first)"""
        with open(schema_path, 'r', encoding='utf-8') as f:
            sql = f.read()
            
        with self.conn.cursor() as cur:
            print("  Dropping old tables (if exists)...")
            # Drop in reverse dependency order
            cur.execute("DROP TABLE IF EXISTS project_embeddings CASCADE;")
            cur.execute("DROP TABLE IF EXISTS project_images CASCADE;")
            cur.execute("DROP TABLE IF EXISTS project_assets CASCADE;")
            cur.execute("DROP TABLE IF EXISTS project_metadata CASCADE;")
            cur.execute("DROP TABLE IF EXISTS projects CASCADE;")
            
            # Also drop old schema tables (from previous version)
            cur.execute("DROP TABLE IF EXISTS embeddings CASCADE;")
            cur.execute("DROP TABLE IF EXISTS images CASCADE;")
            print(" Old tables removed")
            
            print(" Creating new schema...")
            cur.execute(sql)
            print(" Schema V2 Applied Successfully")

    def add_project(self, metadata, text_embedder=None, image_embedder=None, images_dir=None):
        """
        Insert complete project with normalized data
        Args:
            metadata: Dict from metadata.json (standardized format)
            text_embedder: Optional TextEmbedder instance for semantic search
            image_embedder: Optional ImageEmbedder for CLIP vectors
            images_dir: Path object to images folder
        """
        
        # Generate UUID for this project
        proj_uuid = str(uuid.uuid4())
        
        with self.conn.cursor() as cur:
            # 1. Insert Core Project Info
            cur.execute("""
                INSERT INTO projects (id, project_code, title, repo_url, project_type, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (project_code) DO UPDATE
                SET title = EXCLUDED.title,
                    repo_url = EXCLUDED.repo_url,
                    project_type = EXCLUDED.project_type,
                    status = EXCLUDED.status
                RETURNING id;
            """, (
                proj_uuid,
                metadata.get('project_id'),
                metadata.get('title'),
                metadata.get('repo_url'),
                metadata.get('project_type', 'product'),
                metadata.get('status', 'active')
            ))
            
            result = cur.fetchone()
            if result:
                proj_uuid = result[0]  # Use existing ID if conflict
            
            # 2. Insert Project Metadata (1:1)
            cur.execute("""
                INSERT INTO project_metadata (
                    project_id, domain, platform, frontend, backend, database, deployment,
                    estimate_days, complexity, team_size, tags
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (project_id) DO UPDATE
                SET domain = EXCLUDED.domain,
                    platform = EXCLUDED.platform,
                    frontend = EXCLUDED.frontend,
                    backend = EXCLUDED.backend,
                    database = EXCLUDED.database,
                    deployment = EXCLUDED.deployment,
                    estimate_days = EXCLUDED.estimate_days,
                    complexity = EXCLUDED.complexity,
                    team_size = EXCLUDED.team_size,
                    tags = EXCLUDED.tags;
            """, (
                proj_uuid,
                metadata.get('domain', 'other'),
                metadata.get('platform', []),
                metadata.get('frontend', []),
                metadata.get('backend', []),
                metadata.get('database', []),
                metadata.get('deployment', []),
                metadata.get('estimate', {}).get('days', 0),
                metadata.get('estimate', {}).get('complexity', 'medium'),
                metadata.get('team_size', 1),
                metadata.get('tags', [])
            ))
            
            # 3. Insert Text Embeddings (Semantic Documents)
            for doc in metadata.get('semantic_documents', []):
                embedding_type = doc.get('type')
                content = doc.get('content', '')
                
                vector = None
                if text_embedder and content:
                    try:
                        vector = text_embedder.embed(content)  # Fixed: embed() not embed_text()
                    except Exception as e:
                        print(f"  ⚠️ Failed to embed {embedding_type}: {e}")
                        vector = None
                
                cur.execute("""
                    INSERT INTO project_embeddings (project_id, embedding_type, content, embedding)
                    VALUES (%s, %s, %s, %s);
                """, (proj_uuid, embedding_type, content, vector))
            
            # 4. Insert Assets (README, folder structure, etc)
            assets = metadata.get('assets', {})
            
            # README
            if assets.get('readme'):
                cur.execute("""
                    INSERT INTO project_assets (project_id, asset_type, content)
                    VALUES (%s, 'readme', %s);
                """, (proj_uuid, assets['readme']))
            
            # Folder Structure
            if assets.get('folder_structure'):
                folder_json = json.dumps(assets['folder_structure'])
                cur.execute("""
                    INSERT INTO project_assets (project_id, asset_type, content)
                    VALUES (%s, 'folder_structure', %s);
                """, (proj_uuid, folder_json))
            
            # 5. Insert Images with Embeddings (if provided)
            if image_embedder and images_dir and images_dir.exists():
                image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
                
                for img_path in image_files:
                    try:
                        # Generate CLIP embedding and flatten to 1D
                        vector_raw = image_embedder.embed_image(str(img_path))
                        vector = vector_raw.flatten().tolist()  # Fix: Ensure 1D array
                        
                        cur.execute("""
                            INSERT INTO project_images (project_id, image_name, image_path, embedding)
                            VALUES (%s, %s, %s, %s);
                        """, (proj_uuid, img_path.name, str(img_path), vector))
                    except Exception as e:
                        print(f"  ⚠️ Failed to embed image {img_path.name}: {e}")
        
        return proj_uuid

    def search_projects(self, query_vector=None, filters=None, limit=10):
        """
        Hybrid search with new schema
        Args:
            query_vector: 384-dim vector for semantic search
            filters: Dict with keys like 'domain', 'frontend', 'backend', 'tags'
        """
        
        sql = """
            SELECT DISTINCT p.id, p.title, p.project_code, p.repo_url,
                   pm.domain, pm.platform, pm.frontend, pm.backend,
                   pm.complexity, pm.team_size, pm.tags
        """
        
        if query_vector:
            sql += ", (pe.embedding <=> %s::vector) as distance "
            sql += """
                FROM projects p
                JOIN project_metadata pm ON p.id = pm.project_id
                LEFT JOIN project_embeddings pe ON p.id = pe.project_id 
                    AND pe.embedding_type = 'description'
                WHERE 1=1
            """
            params = [query_vector]
        else:
            sql += """
                FROM projects p
                JOIN project_metadata pm ON p.id = pm.project_id
                WHERE 1=1
            """
            params = []
        
        # Apply filters
        if filters:
            if filters.get('domain'):
                sql += " AND pm.domain = %s"
                params.append(filters['domain'])
            
            if filters.get('frontend'):
                sql += " AND pm.frontend && %s"
                params.append(filters['frontend'])
            
            if filters.get('backend'):
                sql += " AND pm.backend && %s"
                params.append(filters['backend'])
            
            if filters.get('tags'):
                sql += " AND pm.tags && %s"
                params.append(filters['tags'])
            
            if filters.get('complexity'):
                sql += " AND pm.complexity = %s"
                params.append(filters['complexity'])
        
        # Order and limit
        if query_vector:
            sql += " ORDER BY distance ASC NULLS LAST"
        else:
            sql += " ORDER BY p.title"
        
        sql += " LIMIT %s"
        params.append(limit)
        
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            results = cur.fetchall()
            
            # Calculate score boost based on refined filters
            # If user provided specific filters, we are more confident in the result
            filter_boost = 0.0
            if filters:
                # Boost 0.1 for each meaningful filter applied
                filter_boost += 0.1 * len([k for k, v in filters.items() if v])
                
            return [
                {
                    "id": r[0],
                    "title": r[1],
                    "project_code": r[2],
                    "repo_url": r[3],
                    "domain": r[4],
                    "platform": r[5],
                    "frontend": r[6],
                    "backend": r[7],
                    "complexity": r[8],
                    "team_size": r[9],
                    "tags": r[10],
                    "score": min((1 - r[11]) + filter_boost, 0.99) if query_vector and len(r) > 11 and r[11] is not None else filter_boost
                }
                for r in results
            ]

    def count_projects(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM projects;")
            return cur.fetchone()[0]
    
    def count_images(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM project_images;")
            return cur.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()
