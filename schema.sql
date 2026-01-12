-- =============================================================================
-- SCHEMA FOR COMPONENT-BASED IMAGE SEARCH SYSTEM
-- =============================================================================
-- This schema supports the following workflow:
-- 1. User sends an image → Agent detects UI components
-- 2. Each component embedding queries vector DB
-- 3. Returns similar components with source_code file references
-- 4. Agent fetches code from GitHub using repo_url + file_path + line range
-- =============================================================================

-- Enable pgvector extension for storing embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- TABLE 1: PROJECTS
-- Core project information with GitHub repository reference
-- =============================================================================
CREATE TABLE IF NOT EXISTS projects (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_code  TEXT UNIQUE NOT NULL,        -- e.g., "project_008_paypal"
    title         TEXT NOT NULL,               -- e.g., "PayPal Clone"
    repo_url      TEXT NOT NULL,               -- GitHub repo URL for fetching code
    domain        TEXT,                        -- fintech, ecommerce, social, etc.
    
    -- Tech Stack (for filtering)
    frontend      TEXT[],                      -- ["React", "TypeScript"]
    backend       TEXT[],                      -- ["Node.js", "Express"]
    database      TEXT[],                      -- ["PostgreSQL", "Redis"]
    
    created_at    TIMESTAMP DEFAULT now(),
    updated_at    TIMESTAMP DEFAULT now()
);

-- =============================================================================
-- TABLE 2: PROJECT IMAGES
-- Screenshots/mockups belonging to a project
-- =============================================================================
CREATE TABLE IF NOT EXISTS project_images (
    id            SERIAL PRIMARY KEY,
    project_id    UUID NOT NULL
                  REFERENCES projects(id) ON DELETE CASCADE,
    
    image_id      TEXT UNIQUE,                 -- e.g., "paypal_img_001"
    image_path    TEXT NOT NULL,               -- Relative path: "images/image1.png"
    page_name     TEXT,                        -- e.g., "Homepage - Hero Section"
    
    -- Full image embedding for image-level search
    embedding     VECTOR(512),                 -- CLIP ViT-B/32 embedding
    
    created_at    TIMESTAMP DEFAULT now()
);

-- =============================================================================
-- TABLE 3: COMPONENTS (Core table for component-level search)
-- Individual UI components detected from images with source code references
-- =============================================================================
CREATE TABLE IF NOT EXISTS components (
    id              SERIAL PRIMARY KEY,
    image_id        INT NOT NULL
                    REFERENCES project_images(id) ON DELETE CASCADE,
    project_id      UUID NOT NULL
                    REFERENCES projects(id) ON DELETE CASCADE,
    
    -- Component Identity
    component_id    TEXT UNIQUE,               -- e.g., "paypal_comp_001"
    component_type  TEXT NOT NULL,             -- header, hero, login-form, etc.
    component_name  TEXT,                      -- "Navigation Header"
    
    -- Semantic Information
    semantic_tags   TEXT[],                    -- ["navigation", "logo", "menu"]
    description     TEXT,                      -- Human-readable description
    
    -- Source Code Reference (Key for fetching code from GitHub)
    source_file_path    TEXT,                  -- "src/components/Header/Header.tsx"
    source_start_line   INT,                   -- 1
    source_end_line     INT,                   -- 55
    
    -- Visual Information
    bbox            JSONB,                     -- {"x": 0, "y": 0, "w": 1920, "h": 80}
    bbox_norm       JSONB,                     -- {"x": 0, "y": 0, "w": 1.0, "h": 0.05}
    
    -- Component Embedding for Vector Search
    embedding       VECTOR(512),               -- CLIP embedding of cropped component
    confidence      FLOAT DEFAULT 1.0,         -- Detection confidence (0-1)
    
    created_at      TIMESTAMP DEFAULT now(),
    updated_at      TIMESTAMP DEFAULT now()
);

-- =============================================================================
-- INDEXES FOR HIGH PERFORMANCE
-- =============================================================================

-- Project Filtering
CREATE INDEX IF NOT EXISTS idx_projects_domain ON projects(domain);
CREATE INDEX IF NOT EXISTS idx_projects_frontend ON projects USING GIN(frontend);
CREATE INDEX IF NOT EXISTS idx_projects_backend ON projects USING GIN(backend);

-- Image Lookup
CREATE INDEX IF NOT EXISTS idx_images_project ON project_images(project_id);

-- Component Filtering
CREATE INDEX IF NOT EXISTS idx_components_project ON components(project_id);
CREATE INDEX IF NOT EXISTS idx_components_image ON components(image_id);
CREATE INDEX IF NOT EXISTS idx_components_type ON components(component_type);
CREATE INDEX IF NOT EXISTS idx_components_tags ON components USING GIN(semantic_tags);

-- Vector Similarity Search (HNSW for fast approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_image_embedding
ON project_images USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_component_embedding
ON components USING hnsw (embedding vector_cosine_ops);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to get GitHub raw URL for fetching source code
CREATE OR REPLACE FUNCTION get_source_url(p_component_id TEXT)
RETURNS TABLE (
    raw_url TEXT,
    start_line INT,
    end_line INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        REPLACE(p.repo_url, 'github.com', 'raw.githubusercontent.com') 
            || '/main/' || c.source_file_path AS raw_url,
        c.source_start_line AS start_line,
        c.source_end_line AS end_line
    FROM components c
    JOIN projects p ON c.project_id = p.id
    WHERE c.component_id = p_component_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- EXAMPLE QUERIES
-- =============================================================================

-- 1. Find similar components by embedding (vector search)
-- SELECT c.*, p.repo_url,
--        1 - (c.embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
-- FROM components c
-- JOIN projects p ON c.project_id = p.id
-- WHERE c.component_type = 'header'
-- ORDER BY c.embedding <=> '[0.1, 0.2, ...]'::vector
-- LIMIT 5;

-- 2. Get source code info for a matched component
-- SELECT p.repo_url, c.source_file_path, c.source_start_line, c.source_end_line
-- FROM components c
-- JOIN projects p ON c.project_id = p.id
-- WHERE c.component_id = 'paypal_comp_002';

-- 3. Filter by semantic tags
-- SELECT * FROM components
-- WHERE semantic_tags @> ARRAY['navigation', 'header']
-- ORDER BY created_at DESC;

-- =============================================================================
-- ANALYZE FOR QUERY OPTIMIZATION
-- =============================================================================
ANALYZE projects;
ANALYZE project_images;
ANALYZE components;
