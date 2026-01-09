"""
Visual Comparison Generator
Tạo ảnh so sánh query image với matched components
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

sys.path.append(os.getcwd())

from src.postgres_db import PostgresDB
from src.component_detector import UIComponentDetector
from src.component_embedder import ComponentEmbedder


def create_comparison_image(query_path, matches_data, output_path="demo_visual_output.jpg"):
    """
    Tạo ảnh comparison ĐƠNG GIẢN - chỉ component quan trọng
    """
    # Load query image
    query_img = cv2.imread(query_path)
    orig_h, orig_w = query_img.shape[:2]
    
    # Scale if too large and TRACK scale factor
    max_width = 700
    scale_factor = 1.0
    if orig_w > max_width:
        scale_factor = max_width / orig_w
        query_img = cv2.resize(query_img, (int(orig_w*scale_factor), int(orig_h*scale_factor)))
    
    h, w = query_img.shape[:2]
    
    # Create canvas
    canvas_width = w * 2 + 100
    canvas_height = max(h + 150, 900)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 250
    
    # === LEFT SIDE: Paste Query Image FIRST ===
    query_y_offset = 80
    canvas[query_y_offset:query_y_offset+h, 20:20+w] = query_img
    
    # Professional colors (bright and clear)
    colors = {
        'header': (255, 140, 0),       # Orange
        'footer': (0, 200, 0),         # Green  
        'cta_button': (0, 0, 255),     # Red (BGR)
        'form': (255, 200, 0),         # Cyan
        'login_form': (255, 200, 0),
        'section': (200, 100, 255),    # Purple
        'navigation': (200, 150, 100),
        'sidebar': (180, 100, 200),
    }
    
    # FILTER: Only important types
    important_types = ['header', 'footer', 'cta_button', 'form', 'login_form', 
                       'navigation', 'section', 'sidebar']
    
    # === Draw boxes ON TOP of query image with numbers ===
    print("Drawing boxes on query image...")
    drawn_count = 0
    component_numbers = {}  # Track numbers for each type
    
    for match_group in matches_data:
        comp_type = match_group['type']
        
        if comp_type not in important_types:
            continue
        
        input_comp = match_group['input_component']
        bbox = input_comp['bbox']
        color = colors.get(comp_type, (120, 120, 120))
        
        # SCALE bbox coordinates to match resized image
        x = int(bbox[0] * scale_factor)
        y = int(bbox[1] * scale_factor)
        bw = int(bbox[2] * scale_factor)
        bh = int(bbox[3] * scale_factor)
        
        # Assign number
        drawn_count += 1
        component_numbers[comp_type] = drawn_count
        
        # Draw thick rectangle
        cv2.rectangle(canvas, 
                     (20 + x, query_y_offset + y), 
                     (20 + x + bw, query_y_offset + y + bh),
                     color, 4)
        
        # NUMBER BADGE on top-left corner
        badge_x = 20 + x + 10
        badge_y = query_y_offset + y + 25
        
        # White circle background
        cv2.circle(canvas, (badge_x, badge_y), 18, (255, 255, 255), -1)
        # Colored circle
        cv2.circle(canvas, (badge_x, badge_y), 16, color, -1)
        # Number
        cv2.putText(canvas, str(drawn_count), 
                   (badge_x - 8 if drawn_count < 10 else badge_x - 12, badge_y + 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Label with solid background
        label = comp_type.replace('_', ' ').upper()
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Label background
        cv2.rectangle(canvas, 
                     (20 + x, query_y_offset + y - text_h - 12),
                     (20 + x + text_w + 12, query_y_offset + y - 2),
                     color, -1)
        
        # Label text
        cv2.putText(canvas, label, 
                   (20 + x + 6, query_y_offset + y - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        print(f"  ✓ [{drawn_count}] {comp_type} at [{x},{y},{bw},{bh}]")
    
    print(f"Total boxes drawn: {drawn_count}")
    
    # Store for right side
    match_numbers = component_numbers
    
    # Label for query
    cv2.putText(canvas, "QUERY IMAGE", 
               (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    
    # === RIGHT SIDE: Match Results WITH THUMBNAILS ===
    right_x = w + 60
    y_offset = 80
    
    cv2.putText(canvas, "TOP MATCHES",
               (right_x, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
    
    displayed = 0
    for match_group in matches_data:
        comp_type = match_group['type']
        
        if comp_type not in important_types:
            continue
        
        matches = match_group['matches']
        if not matches:
            continue
        
        # Get matching number
        match_num = match_numbers.get(comp_type, 0)
        
        # Component type header with NUMBER
        color = colors.get(comp_type, (120, 120, 120))
        
        label = comp_type.replace('_', ' ').upper()
        cv2.rectangle(canvas,
                     (right_x, y_offset),
                     (right_x + w - 80, y_offset + 35),
                     color, -1)
        
        # NUMBER BADGE on header
        badge_x = right_x + 25
        badge_y = y_offset + 18
        
        cv2.circle(canvas, (badge_x, badge_y), 16, (255, 255, 255), -1)
        cv2.circle(canvas, (badge_x, badge_y), 14, (50, 50, 50), -1)
        cv2.putText(canvas, str(match_num),
                   (badge_x - 7 if match_num < 10 else badge_x - 11, badge_y + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(canvas, label,
                   (right_x + 55, y_offset + 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y_offset += 45
        
        # Best match
        best = matches[0]
        title, code, img_name, bbox_data, dist = best
        similarity = (1 - dist) * 100
        
        # === LOAD MATCHED IMAGE & CROP COMPONENT ===
        try:
            # Parse bbox - handle list format [x,y,w,h]
            if isinstance(bbox_data, list) and len(bbox_data) == 4:
                mx, my, mw, mh = bbox_data
            else:
                print(f"Invalid bbox format: {bbox_data}")
                raise ValueError("Invalid bbox")
            
            # Try multiple path formats
            possible_paths = [
                f"dataset/{code}/images/{img_name}",
                f"dataset/{code}/{img_name}",
                f"{code}/images/{img_name}",
            ]
            
            matched_img = None
            matched_path = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    matched_img = cv2.imread(path)
                    matched_path = path
                    break
            
            if matched_img is None:
                print(f"Image not found. Tried: {possible_paths[0]}")
                raise FileNotFoundError("Image not found")
            
            # Validate bbox
            img_h, img_w = matched_img.shape[:2]
            if mx < 0 or my < 0 or mx + mw > img_w or my + mh > img_h:
                print(f"Bbox out of bounds: [{mx},{my},{mw},{mh}] for image {img_w}x{img_h}")
                raise ValueError("Bbox out of bounds")
            
            # Crop component
            cropped = matched_img[my:my+mh, mx:mx+mw].copy()
            
            if cropped.size == 0:
                print(f"Empty crop")
                raise ValueError("Empty crop")
            
            # Resize to fit
            max_thumb_w = min(200, w - 100)
            max_thumb_h = 200
            
            if cropped.shape[1] > max_thumb_w or cropped.shape[0] > max_thumb_h:
                scale_w = max_thumb_w / cropped.shape[1]
                scale_h = max_thumb_h / cropped.shape[0]
                scale = min(scale_w, scale_h)
                cropped = cv2.resize(cropped, (int(cropped.shape[1]*scale), int(cropped.shape[0]*scale)))
            
            ch, cw = cropped.shape[:2]
            
            # Check canvas space
            if y_offset + ch + 100 > canvas_height:
                print(f"   Not enough canvas space")
                break
            
            # Place thumbnail
            thumb_y = y_offset
            
            # White background
            cv2.rectangle(canvas,
                         (right_x, thumb_y),
                         (right_x + cw + 10, thumb_y + ch + 10),
                         (255, 255, 255), -1)
            
            # Colored border (SAME as query box)
            cv2.rectangle(canvas,
                         (right_x, thumb_y),
                         (right_x + cw + 10, thumb_y + ch + 10),
                         color, 3)
            
            # Paste cropped image
            canvas[thumb_y+5:thumb_y+5+ch, right_x+5:right_x+5+cw] = cropped
            
            y_offset += ch + 20
            
            print(f"Loaded thumbnail for {comp_type}")
            
        except Exception as e:
            print(f"Could not load thumbnail: {e}")
            # Skip thumbnail, just show text
        
        # Match info below thumbnail
        info_bg_h = 60
        cv2.rectangle(canvas,
                     (right_x, y_offset),
                     (right_x + w - 80, y_offset + info_bg_h),
                     (255, 255, 255), -1)
        
        cv2.rectangle(canvas,
                     (right_x, y_offset),
                     (right_x + w - 80, y_offset + info_bg_h),
                     (220, 220, 220), 1)
        
        # Project name
        cv2.putText(canvas, title, 
                   (right_x + 10, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 2)
        
        # Similarity
        sim_color = (50, 150, 50) if similarity >= 70 else (200, 120, 50)
        cv2.putText(canvas, f"Match: {similarity:.1f}%", 
                   (right_x + 10, y_offset + 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, sim_color, 2)
        
        y_offset += info_bg_h + 15
        displayed += 1
        
        # Limit to 4 to fit in screen
        if displayed >= 4:
            break
    
    # Title bar
    title_height = 70
    title_canvas = np.ones((title_height, canvas_width, 3), dtype=np.uint8) * 60
    cv2.putText(title_canvas, "COMPONENT SEARCH DEMO", 
               (30, 48),
               cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    
    # Combine
    final = np.vstack([title_canvas, canvas])
    
    cv2.imwrite(output_path, final)
    return output_path


def demo_with_visualization(query_path, top_k=3):
    """
    Run search và tạo ảnh visualization
    """
    print("\nCreating visual comparison...")
    
    # 1. Load
    db = PostgresDB()
    detector = UIComponentDetector(method='sam', classify_semantics=True, use_clip=True)
    embedder = ComponentEmbedder()
    
    # 2. Detect
    print(f"Analyzing '{query_path}'...")
    components = detector.detect(query_path)
    components = embedder.embed_components(query_path, components)
    
    # 3. Search - CHỈ component quan trọng
    print("Searching database...")
    
    # FILTER important types
    important_types = {'header', 'footer', 'cta_button', 'form', 'login_form', 
                      'navigation', 'section', 'sidebar', 'search_form'}
    
    matches_data = []
    
    by_type = {}
    for comp in components:
        t = comp.get('semantic_type', 'unknown')
        if t in important_types:  # Only important
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(comp)
    
    for semantic_type, comps in by_type.items():
        query_comp = comps[0]
        query_embedding = query_comp['embedding']
        
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    p.title, p.project_code, pi.image_name, pce.bbox,
                    pce.embedding <=> %s::vector as distance
                FROM project_component_embeddings pce
                JOIN project_images pi ON pce.image_id = pi.id
                JOIN projects p ON pce.project_id = p.id
                WHERE pce.component_type = %s
                ORDER BY pce.embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding, semantic_type, query_embedding, top_k))
            
            matches = cur.fetchall()
        
        if matches:
            matches_data.append({
                'type': semantic_type,
                'input_component': query_comp,
                'matches': matches
            })
    
    # 4. Create visual
    output_path = create_comparison_image(query_path, matches_data)
    
    print(f"\n Visual comparison saved to: {output_path}")
    print(f"Showing {len(matches_data)} important component types\n")
    
    db.close()
    return output_path


if __name__ == "__main__":
    query = "test/test3.jpg"
    
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    if not os.path.exists(query):
        print(f" Error: Image not found: {query}")
        sys.exit(1)
    
    output = demo_with_visualization(query)
    print(f" Open '{output}' to see results!\n")
