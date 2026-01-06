
import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
import uuid
import json

class PostgresDB:
    """
    Quản lý kết nối PostgreSQL cho AI Agent.
    Hỗ trợ lưu trữ structured data (Projects/Images) và Vector (Embeddings) chung 1 chỗ.
    """
    
    def __init__(self, dsn="postgresql://user:password@localhost:5433/vectordb"):
        """
        Args:
            dsn: Connection string (mặc định theo docker-compose.yml)
        """
        self.dsn = dsn
        self.conn = None
        self.connect()
        self.init_schema()
        
    def connect(self):
        try:
            self.conn = psycopg2.connect(self.dsn)
            self.conn.autocommit = True
            # Đăng ký adapter để hiểu kiểu dữ liệu vector
            register_vector(self.conn)
            print("Đã kết nối PostgreSQL thành công!")
        except Exception as e:
            print(f"Lỗi kết nối Postgres: {e}")
            raise e

    def init_schema(self):
        """Khởi tạo bảng theo schema.sql"""
        try:
            with self.conn.cursor() as cur:
                # Đọc file schema.sql nếu cần, hoặc chạy trực tiếp lệnh ở đây
                # Ở đây mình chạy lệnh 'CREATE EXISTENSION' để chắc chắn
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Tạo bảng Projects
                cur.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    tech_stack JSONB DEFAULT '[]',
                    tags JSONB DEFAULT '[]',
                    estimate_days INTEGER,
                    repo_url TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                # Tạo bảng Images
                cur.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id UUID PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    image_name TEXT,
                    image_path TEXT,
                    image_type TEXT DEFAULT 'screenshot',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                # Tạo bảng Embeddings
                cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    image_id UUID PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
                    vector_data VECTOR(512),
                    model_version TEXT DEFAULT 'ViT-B/32'
                );
                """)
                
                # Tạo Index HNSW (để search nhanh)
                cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
                ON embeddings 
                USING hnsw (vector_data vector_cosine_ops);
                """)
                
            print("Đã kiểm tra và khởi tạo Schema.")
            
        except Exception as e:
            print(f"Lỗi khởi tạo schema: {e}")

    def add_project(self, data):
        """Thêm project vào DB"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO projects (id, title, description, tech_stack, tags, estimate_days, repo_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE 
                SET title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    tech_stack = EXCLUDED.tech_stack;
            """, (
                data['project_id'],
                data.get('title', ''),
                data.get('description', ''),
                Json(data.get('tech_stack', [])),
                Json(data.get('tags', [])),
                data.get('estimate_days', 0),
                data.get('repo_url', '')
            ))

    def add_image_with_embedding(self, image_id, project_id, image_path, vector, image_name=""):
        """Thêm ảnh và vector cùng lúc (Transaction)"""
        try:
            with self.conn.cursor() as cur:
                # 1. Thêm ảnh
                cur.execute("""
                    INSERT INTO images (id, project_id, image_name, image_path)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                """, (image_id, project_id, image_name, image_path))
                
                # 2. Thêm vector
                cur.execute("""
                    INSERT INTO embeddings (image_id, vector_data)
                    VALUES (%s, %s)
                    ON CONFLICT (image_id) DO UPDATE SET vector_data = EXCLUDED.vector_data;
                """, (image_id, vector))
                
        except Exception as e:
            print(f"⚠️ Lỗi thêm ảnh {image_name}: {e}")

    def search_similar(self, query_vector, limit=5, filter_tech=None):
        """
        Tìm kiếm Hybrid (Vector + Metadata)
        """
        sql = """
            SELECT p.title, p.repo_url, i.image_name, i.image_path, (e.vector_data <=> %s::vector) as distance
            FROM embeddings e
            JOIN images i ON e.image_id = i.id
            JOIN projects p ON i.project_id = p.id
            WHERE 1=1
        """
        params = [query_vector]
        
        if filter_tech:
            # Ví dụ lọc tech stack: WHERE p.tech_stack @> '["React"]'
            sql += " AND p.tech_stack @> %s"
            params.append(Json([filter_tech]))
            
        sql += " ORDER BY distance ASC LIMIT %s;"
        params.append(limit)
        
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def count_images(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM images;")
            return cur.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()
