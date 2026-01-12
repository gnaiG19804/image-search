
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

LOG_FILE = "internal_log.txt"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

log("Script starting...")

def read_source_code(project, file_path, start_line, end_line):
    """
    Read source code from the dataset directory.
    
    Args:
        project: Project code name (e.g., "project_008_paypal")
        file_path: Relative path to source file (e.g., "src/components/Header/Header.tsx")
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
    
    Returns:
        String containing the code snippet, or error message
    """
    try:
        # Construct full path: dataset/{project}/{file_path}
        # Use forward slashes to match database paths
        full_path = os.path.join("dataset", project, file_path).replace("\\", "/")
        
        if not os.path.exists(full_path):
            return f"[File not found: {full_path}]"
        
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Extract the relevant lines (convert from 1-indexed to 0-indexed)
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        code_lines = lines[start_idx:end_idx]
        return ''.join(code_lines)
    
    except Exception as e:
        return f"[Error reading file: {e}]"

def search_by_image(image_path, top_k=3):
    try:
        log(f"\n==================================================")
        log(f" SEARCHING BY IMAGE: {image_path}")
        log(f"==================================================\n")
        
        if not os.path.exists(image_path):
            log(f"Error: File not found: {image_path}")
            return

        # Lazy imports to show progress
        log("[1/4] Loading AI Models (Importing libs)...")
        import cv2
        import json
        import torch
        import numpy as np
        from src.postgres_db import PostgresDB
        from src.embedding_service import EmbeddingService
        from src.component_detector import UIComponentDetector
        
        log("      -> Libs imported. Initializing classes...")
        db = PostgresDB()
        embedder = EmbeddingService()
        detector = UIComponentDetector() 
        
        # 2. Detect Candidates
        log("\n[2/4] Detecting UI Components in image...")
        # Read image just for dimensions
        image = cv2.imread(image_path)
        if image is None:
            log("Error: Could not read image.")
            return
            
        log(f"      -> Image loaded: {image.shape}")
        
        # Use detector.detect(path)
        candidates = detector.detect(image_path)
        log(f"      -> Found {len(candidates)} potential components.")
        
        # Filter small/garbage candidates
        valid_candidates = []
        img_area = image.shape[0] * image.shape[1]
        
        for c in candidates:
            w, h = c['bbox'][2], c['bbox'][3]
            if (w * h) > (img_area * 0.005):
                valid_candidates.append(c)
                
        log(f"      -> {len(valid_candidates)} valid components after filtering.")
        
        # 3. Process & Query
        log("\n[3/4] Generating Embeddings & Querying Database...")
        
        for i, comp in enumerate(valid_candidates, 1):
            x, y, w, h = comp['bbox'] # consistent unpacking
            crop = comp['image'] # Use pre-cropped image from detector
            
            # Get embedding for this crop
            vector = embedder.get_embedding(crop)
            
            if vector is None:
                continue
                
            # Search DB
            # We search for components that look like this crop
            results = db.search_components(vector, limit=1) # Just get the best match per component
            
            if results:
                best = results[0]
                score = best['score']
                
                # Heuristic: Only show matches with reasonable similarity
                if score > 0.20:
                    log(f"\n   🧩 Component #{i} [Pos: {x},{y} Size: {w}x{h}]")
                    log(f"      ---> MATCH: {best['name']} ({best['type']})")
                    log(f"           Score: {score:.3f}")
                    log(f"           Src:   {best['project']} | {best['file_path']}")
                    log(f"           Code:  L{best['start_line']}-L{best['end_line']}")
                    
                    # Extract and display source code
                    code_snippet = read_source_code(
                        best['project'], 
                        best['file_path'], 
                        best['start_line'], 
                        best['end_line']
                    )
                    
                    log(f"\n           --- SOURCE CODE ---")
                    # Add line numbers to code for readability
                    code_lines = code_snippet.split('\n')
                    for line_num, line_content in enumerate(code_lines, start=best['start_line']):
                        log(f"           {line_num:4d} | {line_content.rstrip()}")
                    log(f"           --- END CODE ---\n")
                else:
                     pass
            
        log("\n[4/4] Done!")
        
    except Exception as e:
        log(f"CRITICAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_by_image.py <path_to_image.png>")
        # Default test
        test_img = "dataset/project_013_amazon/images/image1.png"
        if os.path.exists(test_img):
            search_by_image(test_img)
    else:
        search_by_image(sys.argv[1])
 