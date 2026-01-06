-- Enable pgvector extension for storing embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Projects Table
-- Stores high-level project information
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,                  -- e.g., 'project_001'
    title TEXT NOT NULL,
    description TEXT,
    tech_stack JSONB DEFAULT '[]',        -- Stores array like ["React", "Node"]
    tags JSONB DEFAULT '[]',              -- Stores array like ["web", "mobile"]
    estimate_days INTEGER,
    repo_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Images Table
-- Stores information about individual images belonging to projects
CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY,                  -- Generated UUID
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    image_name TEXT,
    image_path TEXT,                      -- Path in S3 or local storage
    image_type TEXT DEFAULT 'screenshot',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Embeddings Table
-- Stores the 512-dim vector for each image
CREATE TABLE IF NOT EXISTS embeddings (
    image_id UUID PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    vector_data VECTOR(512),              -- CLIP ViT-B/32 uses 512 dimensions
    model_version TEXT DEFAULT 'ViT-B/32'
);

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
ON embeddings 
USING hnsw (vector_data vector_cosine_ops);
