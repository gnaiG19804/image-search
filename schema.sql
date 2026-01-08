-- Enable pgvector extension for storing embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Projects Table
-- Stores high-level project information (Core Identity)
CREATE TABLE IF NOT EXISTS projects (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_code  TEXT UNIQUE,                 
    title         TEXT NOT NULL,
    repo_url      TEXT,
    project_type  TEXT CHECK (project_type IN ('product','internal','poc','demo')),
    status        TEXT CHECK (status IN ('active','archived','deprecated')),
    created_at    TIMESTAMP DEFAULT now()
);


-- 2. project_metadata
-- 1:1 Relationship - Stores queryable attributes
CREATE TABLE IF NOT EXISTS project_metadata (
    project_id    UUID PRIMARY KEY
                  REFERENCES projects(id) ON DELETE CASCADE,

    domain        TEXT NOT NULL,          -- ecommerce, fintech, social
    platform      TEXT[] NOT NULL,        -- web, mobile
    frontend      TEXT[] NOT NULL,        -- react, react-native
    backend       TEXT[],                 -- nodejs, spring
    database      TEXT[],                 -- postgres, mongodb
    deployment    TEXT[],                 -- cloud, onprem

    estimate_days INT,
    complexity    TEXT CHECK (complexity IN ('low','medium','high')),
    team_size     INT,

    tags          TEXT[]
);

-- 3. project_embeddings
-- 1:N Relationship - Stores Semantic Vectors (Text)
CREATE TABLE IF NOT EXISTS project_embeddings (
    id             SERIAL PRIMARY KEY,
    project_id     UUID
                   REFERENCES projects(id) ON DELETE CASCADE,

    embedding_type TEXT NOT NULL,      -- description | tech_stack | readme | code_summary
    content        TEXT NOT NULL,      -- The chunk text used for embedding

    embedding      VECTOR(384)         -- MiniLM / BGE
);


-- 4. project_images
-- 1:N Relationship - Stores Visual Vectors (CLIP)
CREATE TABLE IF NOT EXISTS project_images (
    id                   SERIAL PRIMARY KEY,
    project_id           UUID
                         REFERENCES projects(id) ON DELETE CASCADE,

    image_name           TEXT,
    image_path           TEXT,

    embedding            VECTOR(512),           -- CLIP (ViT-B/32)
    components_metadata  JSONB                  -- Detected UI components metadata
    -- Structure: {
    --   "total_components": 5,
    --   "by_type": {
    --     "header": {"count": 1, "avg_size": 0.15},
    --     "hero": {"count": 1, "avg_size": 0.45}
    --   },
    --   "layout_signature": "header-hero-gallery-footer",
    --   "components": [
    --     {
    --       "type": "header",
    --       "semantic_type": "header",
    --       "bbox": [0, 0, 1920, 100],
    --       "bbox_norm": [0, 0, 1, 0.05],
    --       "confidence": 0.95
    --     }
    --   ]
    -- }
);

-- 5. project_assets
-- 1:N Relationship - Stores Blob Content (Docs)
CREATE TABLE IF NOT EXISTS project_assets (
    id           SERIAL PRIMARY KEY,
    project_id   UUID
                 REFERENCES projects(id) ON DELETE CASCADE,

    asset_type   TEXT,                  -- readme | package_json | folder_tree
    content      TEXT
);


-- 6. project_component_embeddings
-- 1:N Relationship - Stores Component-Level Visual Vectors
-- Enables component-based search (e.g., find similar hero sections)
CREATE TABLE IF NOT EXISTS project_component_embeddings (
    id                SERIAL PRIMARY KEY,
    image_id          INT
                      REFERENCES project_images(id) ON DELETE CASCADE,
    project_id        UUID
                      REFERENCES projects(id) ON DELETE CASCADE,
    
    component_type    TEXT NOT NULL,       -- header, hero, gallery, footer, CTA, form, etc.
    component_index   INT NOT NULL,        -- Order of component in the image (0-based)
    
    bbox              JSONB,               -- Bounding box: {"x": 0, "y": 0, "w": 1920, "h": 100}
    bbox_norm         JSONB,               -- Normalized bbox: {"x": 0, "y": 0, "w": 1, "h": 0.05}
    
    embedding         VECTOR(512),         -- CLIP embedding of cropped component
    confidence        FLOAT                -- Semantic classification confidence (0-1)
);


-- Create Indexes for High Performance

-- Metadata Filtering (GIN is perfect for Array overlap @>)
CREATE INDEX IF NOT EXISTS idx_meta_domain
ON project_metadata(domain);

CREATE INDEX IF NOT EXISTS idx_meta_platform
ON project_metadata USING GIN(platform);

CREATE INDEX IF NOT EXISTS idx_meta_frontend
ON project_metadata USING GIN(frontend);

CREATE INDEX IF NOT EXISTS idx_meta_backend
ON project_metadata USING GIN(backend);

-- Vector Similarity Search (HNSW is faster/better recall than IVFFlat for production)
CREATE INDEX IF NOT EXISTS idx_text_embedding
ON project_embeddings
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_image_embedding
ON project_images
USING hnsw (embedding vector_cosine_ops);

-- Component Metadata JSONB Queries
CREATE INDEX IF NOT EXISTS idx_components_metadata
ON project_images USING GIN(components_metadata);

-- Component-Level Indexes
CREATE INDEX IF NOT EXISTS idx_component_type
ON project_component_embeddings(component_type);

CREATE INDEX IF NOT EXISTS idx_component_image
ON project_component_embeddings(image_id);

CREATE INDEX IF NOT EXISTS idx_component_embedding
ON project_component_embeddings
USING hnsw (embedding vector_cosine_ops);


ANALYZE;
