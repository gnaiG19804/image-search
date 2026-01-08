
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

class UIComponentDetector:
    """
    Phát hiện các thành phần UI trong screenshot
    - rule_based: Phân chia cứng theo tỷ lệ (nhanh, không cần GPU)
    - sam: Tự động phát hiện mọi object bằng AI (chính xác, cần GPU)
    """
    
    def __init__(self, method='sam', sam_model_type='vit_b', classify_semantics=False, use_clip=False):
        """
        Args:
            method: 'rule_based' hoặc 'sam' (recommended)
            sam_model_type: 'vit_b' (default, 375MB) | 'vit_l' (1.2GB) | 'vit_h' (2.4GB)
            classify_semantics: Enable semantic classification (header, hero, CTA, etc.)
            use_clip: Use CLIP for semantic classification (requires CLIP installed)
        """
        self.method = method
        self.sam_model = None
        self.mask_generator = None
        self.classify_semantics = classify_semantics
        self.semantic_classifier = None
        
        if method == 'sam':
            self._init_sam(sam_model_type)
        
        # Initialize semantic classifier if requested
        if classify_semantics:
            from src.semantic_classifier import SemanticClassifier
            self.semantic_classifier = SemanticClassifier(use_clip=use_clip)
            print(f"[INFO] Semantic classifier initialized (CLIP: {use_clip})")
        
        # Thresholds cho rule-based detection (fallback)
        self.header_ratio = 0.15
        self.footer_ratio = 0.10
        self.min_card_width = 100
        self.min_card_height = 100
    
    def _init_sam(self, model_type):
        """Initialize SAM (Segment Anything Model)"""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            import torch
            
            # Download URLs for SAM checkpoints
            model_urls = {
                'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
                'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
                'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
            }
            
            # Path to save/load checkpoint
            checkpoint_dir = Path('models/sam')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f'sam_{model_type}.pth'
            
            # Download if not exists
            if not checkpoint_path.exists():
                print(f"[INFO] Downloading SAM checkpoint ({model_type})...")
                import urllib.request
                urllib.request.urlretrieve(model_urls[model_type], checkpoint_path)
                print(f"[INFO] Downloaded to {checkpoint_path}")
            
            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[INFO] Loading SAM model on {device}...")
            
            sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
            sam.to(device=device)
            
            # Create automatic mask generator (RELAXED for better coverage)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,  # Grid resolution
                pred_iou_thresh=0.75,  # 0.86 → 0.75 (detect more elements)
                stability_score_thresh=0.85,  # 0.92 → 0.85 (allow less stable objects)
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=50,  # 100 → 50 (catch smaller buttons/icons)
            )
            
            print(f"[INFO] SAM loaded successfully!")
            
        except ImportError:
            print("[ERROR] segment-anything not installed!")
            print("Install: pip install git+https://github.com/facebookresearch/segment-anything.git")
            raise
    
        
    def detect(self, image_path: str) -> List[Dict]:
        """
        Phát hiện components trong một ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            List of dictionaries chứa:
            {
                'type': 'header'|'body'|'footer'|'card',
                'bbox': [x, y, width, height] (pixel coordinates),
                'bbox_norm': [x, y, w, h] (normalized 0-1),
                'image': numpy array của component
            }
        """
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        if self.method == 'rule_based':
            return self._detect_rule_based(img)
        elif self.method == 'sam':
            return self._detect_sam(img)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _detect_sam(self, img: np.ndarray) -> List[Dict]:
        """
        SAM-based automatic detection: Tự động phát hiện mọi object trong ảnh
        """
        if self.mask_generator is None:
            raise RuntimeError("SAM model not initialized! Use method='sam' in __init__")
        
        h, w = img.shape[:2]
        components = []
        
        # Generate masks automatically
        print(f"[INFO] Running SAM inference...")
        masks = self.mask_generator.generate(img)
        print(f"[INFO] SAM detected {len(masks)} objects")
        
        # Convert masks to components
        for idx, mask_data in enumerate(masks):
            # Extract segmentation mask
            mask = mask_data['segmentation']  # Boolean array
            
            # Get bounding box
            bbox_xywh = mask_data['bbox']  # [x, y, w, h]
            x, y, w_box, h_box = [int(v) for v in bbox_xywh]
            
            # Crop component using mask
            try:
                # Create empty crop
                crop = np.zeros((h_box, w_box, 3), dtype=np.uint8)
                
                # Extract region
                region = img[y:y+h_box, x:x+w_box]
                region_mask = mask[y:y+h_box, x:x+w_box]
                
                # Apply mask
                crop[region_mask] = region[region_mask]
                
                # Classify component type based on position
                comp_type = self._classify_component_type(y, h, w_box, h_box)
                
                components.append({
                    'type': comp_type,
                    'bbox': [x, y, w_box, h_box],
                    'bbox_norm': [
                        x / w,
                        y / h,
                        w_box / w,
                        h_box / h
                    ],
                    'image': crop,
                    'confidence': float(mask_data.get('predicted_iou', 0.0))
                })
                
            except Exception as e:
                print(f"[WARN] Failed to process mask {idx}: {e}")
                continue
        
        # Post-processing: Filter và clean up
        print(f"[INFO] Filtering components...")
        components = self._filter_components(components, img.shape)
        print(f"[INFO] After filtering: {len(components)} elements detected")
        
        # DISABLED: Hierarchical grouping (user wants all raw elements)
        # print(f"[INFO] Grouping into sections...")
        # components = self._hierarchical_grouping(components, img.shape)
        # print(f"[INFO] After grouping: {len(components)} sections")
        
        # Semantic classification (if enabled)
        if self.classify_semantics and self.semantic_classifier:
            print(f"[INFO] Classifying semantic types...")
            components = self.semantic_classifier.classify_all(components, (h, w))
            # Count by type
            type_counts = {}
            for comp in components:
                comp_type = comp.get('semantic_type', 'unknown')
                type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
            print(f"[INFO] Semantic types detected: {dict(sorted(type_counts.items()))}")
        
        return components
    
    def _filter_components(self, components: List[Dict], img_shape) -> List[Dict]:
        """
        Lọc và làm sạch components để loại bỏ nhiễu
        """
        h, w = img_shape[:2]
        img_area = h * w
        
        filtered = []
        
        for comp in components:
            bbox = comp['bbox']
            comp_w, comp_h = bbox[2], bbox[3]
            comp_area = comp_w * comp_h
            
            # Filter 1: Loại bỏ objects CỰC nhỏ (< 0.05% diện tích ảnh) - VERY RELAXED
            if comp_area < img_area * 0.0005:  # Only remove tiny noise
                continue
            
            # Filter 2: Loại bỏ objects quá to (> 85% - background)
            if comp_area > img_area * 0.85:
                continue
            
            # Filter 3: Loại bỏ confidence CỰC thấp - VERY RELAXED
            if comp.get('confidence', 0) < 0.70:  # 0.75 → 0.70
                continue
            
            # Filter 4: Loại bỏ objects quá dài/hẹp (aspect ratio > 15)
            aspect_ratio = max(comp_w, comp_h) / (min(comp_w, comp_h) + 1e-6)
            if aspect_ratio > 15:  # 10 → 15 (allow longer elements)
                continue
            
            filtered.append(comp)
        
        # Filter 5: Non-Maximum Suppression với threshold cao hơn (giữ nhiều boxes hơn)
        filtered = self._non_max_suppression(filtered, iou_threshold=0.7)  # 0.5 → 0.7
        
        # REMOVED: Top-K limit - Return ALL detected elements
        # User wants to see everything SAM can detect
        print(f"[INFO] Total elements after filtering: {len(filtered)}")
        
        return filtered
    
    def _non_max_suppression(self, components: List[Dict], iou_threshold=0.5) -> List[Dict]:
        """
        Loại bỏ các boxes trùng lặp/overlap nhiều
        """
        if not components:
            return []
        
        # Sort by confidence descending
        components = sorted(components, key=lambda c: c.get('confidence', 0), reverse=True)
        
        keep = []
        
        for comp in components:
            # Check overlap với các boxes đã giữ lại
            should_keep = True
            
            for kept_comp in keep:
                iou = self._calculate_iou(comp['bbox'], kept_comp['bbox'])
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(comp)
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) của 2 bounding boxes
        box format: [x, y, width, height]
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def _hierarchical_grouping(self, components: List[Dict], img_shape) -> List[Dict]:
        """
        Group nearby components into meaningful UI sections
        Sử dụng spatial proximity clustering với logic thông minh
        """
        if not components:
            return []
        
        h, w = img_shape[:2]
        
        # Step 1: Phân chia theo vertical regions (smarter logic)
        regions = {
            'header': [],
            'body': [],
            'footer_candidates': []
        }
        
        for comp in components:
            y_center = comp['bbox'][1] + comp['bbox'][3] / 2
            y_ratio = y_center / h
            bbox_w = comp['bbox'][2]
            
            # Header: Top 25% (mở rộng hơn để catch navbar)
            if y_ratio < 0.25:
                regions['header'].append(comp)
            # Footer candidates: Chỉ ở rất bottom (>95%) VÀ rộng (>70% width)
            elif y_ratio > 0.95 and bbox_w > w * 0.7:
                regions['footer_candidates'].append(comp)
            # Everything else is body
            else:
                regions['body'].append(comp)
        
        # Step 2: Validate footer (chỉ tạo footer nếu thực sự có full-width section ở bottom)
        has_real_footer = False
        if regions['footer_candidates']:
            # Check if any component spans almost full width
            for comp in regions['footer_candidates']:
                if comp['bbox'][2] > w * 0.9:
                    has_real_footer = True
                    break
        
        # If no real footer, move candidates to body
        if not has_real_footer:
            regions['body'].extend(regions['footer_candidates'])
            regions['footer_candidates'] = []
        
        # Step 3: Build final groups
        grouped = []
        
        # Header: Merge nếu có
        if regions['header']:
            header_comp = self._merge_components(regions['header'], 'header', w, h)
            # Validate: Header phải ở top 30% của ảnh
            if header_comp['bbox'][1] < h * 0.3:
                grouped.append(header_comp)
            else:
                # False positive, add to body
                regions['body'].extend(regions['header'])
        
        # Footer: Merge nếu validated
        if regions['footer_candidates']:
            grouped.append(self._merge_components(regions['footer_candidates'], 'footer', w, h))
        
        # Body: Cluster thành cards/sections
        if regions['body']:
            body_sections = self._cluster_components(regions['body'], w, h)
            grouped.extend(body_sections)
        
        return grouped
    
    def _cluster_components(self, components: List[Dict], img_width, img_height) -> List[Dict]:
        """
        Cluster components thành groups dựa trên khoảng cách spatial
        """
        if not components:
            return []
        
        # Distance threshold (% of image dimension)
        distance_threshold = min(img_width, img_height) * 0.15
        
        # Simple agglomerative clustering
        clusters = [[comp] for comp in components]
        
        merged = True
        while merged:
            merged = False
            new_clusters = []
            used = set()
            
            for i, cluster1 in enumerate(clusters):
                if i in used:
                    continue
                
                # Try to merge with other clusters
                for j, cluster2 in enumerate(clusters):
                    if i >= j or j in used:
                        continue
                    
                    # Calculate distance between cluster centroids
                    dist = self._cluster_distance(cluster1, cluster2)
                    
                    if dist < distance_threshold:
                        # Merge clusters
                        cluster1.extend(cluster2)
                        used.add(j)
                        merged = True
                
                new_clusters.append(cluster1)
                used.add(i)
            
            clusters = new_clusters
            
            if not merged:
                break
        
        # Convert clusters to merged components
        sections = []
        for cluster in clusters:
            merged_comp = self._merge_components(cluster, 'card', img_width, img_height)
            sections.append(merged_comp)
        
        return sections
    
    def _cluster_distance(self, cluster1: List[Dict], cluster2: List[Dict]) -> float:
        """
        Calculate distance between two clusters (min distance between any two points)
        """
        min_dist = float('inf')
        
        for c1 in cluster1:
            for c2 in cluster2:
                dist = self._component_distance(c1, c2)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _component_distance(self, comp1: Dict, comp2: Dict) -> float:
        """
        Euclidean distance between component centers
        """
        x1, y1, w1, h1 = comp1['bbox']
        x2, y2, w2, h2 = comp2['bbox']
        
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2
        
        return ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
    
    def _merge_components(self, components: List[Dict], section_type: str, img_w=None, img_h=None) -> Dict:
        """
        Merge multiple components thành 1 bounding box lớn
        """
        if not components:
            return None
        
        # Find bounding box that covers all components
        min_x = min(c['bbox'][0] for c in components)
        min_y = min(c['bbox'][1] for c in components)
        max_x = max(c['bbox'][0] + c['bbox'][2] for c in components)
        max_y = max(c['bbox'][1] + c['bbox'][3] for c in components)
        
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        
        # Average confidence
        avg_conf = sum(c.get('confidence', 0) for c in components) / len(components)
        
        # Get sample image from first component (placeholder)
        sample_image = components[0].get('image', np.zeros((bbox_h, bbox_w, 3), dtype=np.uint8))
        
        # Use actual dimensions or fallback to first component's norm
        if img_w and img_h:
            bbox_norm = [
                min_x / img_w,
                min_y / img_h,
                bbox_w / img_w,
                bbox_h / img_h
            ]
        else:
            bbox_norm = components[0].get('bbox_norm', [0, 0, 1, 1])
        
        return {
            'type': section_type,
            'bbox': [min_x, min_y, bbox_w, bbox_h],
            'bbox_norm': bbox_norm,
            'image': sample_image,
            'confidence': avg_conf,
            'num_children': len(components)  # Track how many were merged
        }
    
    def _classify_component_type(self, y, img_height, width, height):
        """
        Phân loại component type dựa vào vị trí trong ảnh
        """
        y_ratio = y / img_height
        
        # Header: Top 20%
        if y_ratio < 0.2:
            return 'header'
        # Footer: Bottom 15%
        elif y_ratio > 0.85:
            return 'footer'
        # Card: Rectangular shape in middle
        elif width > 150 and height > 150:
            return 'card'
        # Default: body element
        else:
            return 'element'
    
    def _detect_rule_based(self, img: np.ndarray) -> List[Dict]:
        """
        Rule-based detection: header/footer theo tỉ lệ, cards bằng edge detection
        """
        h, w = img.shape[:2]
        components = []
        
        # 1. Header (Top 15%)
        header_h = int(h * self.header_ratio)
        if header_h > 50:  # Chỉ tạo header nếu đủ lớn
            header_img = img[0:header_h, :]
            components.append({
                'type': 'header',
                'bbox': [0, 0, w, header_h],
                'bbox_norm': [0.0, 0.0, 1.0, self.header_ratio],
                'image': header_img
            })
        
        # 2. Footer (Bottom 10%)
        footer_start = int(h * (1 - self.footer_ratio))
        footer_h = h - footer_start
        if footer_h > 50:
            footer_img = img[footer_start:h, :]
            components.append({
                'type': 'footer',
                'bbox': [0, footer_start, w, footer_h],
                'bbox_norm': [0.0, 1 - self.footer_ratio, 1.0, self.footer_ratio],
                'image': footer_img
            })
        
        # 3. Body (Middle section)
        body_start = header_h
        body_end = footer_start
        body_img = img[body_start:body_end, :]
        
        if body_end > body_start:
            components.append({
                'type': 'body',
                'bbox': [0, body_start, w, body_end - body_start],
                'bbox_norm': [0.0, self.header_ratio, 1.0, 1 - self.header_ratio - self.footer_ratio],
                'image': body_img
            })
            
            # 4. Detect cards/sections in body using edge detection
            cards = self._detect_cards_in_region(body_img, body_start, w, h)
            components.extend(cards)
        
        return components
    
    def _detect_cards_in_region(self, region_img: np.ndarray, y_offset: int, 
                                  img_width: int, img_height: int) -> List[Dict]:
        """
        Phát hiện các "cards" (sections có viền rõ ràng) trong một vùng
        
        Args:
            region_img: Vùng ảnh cần detect (ví dụ body)
            y_offset: Offset y của vùng này so với ảnh gốc
            img_width, img_height: Kích thước ảnh gốc để normalize
        """
        cards = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate để làm nổi bật các viền
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter: chỉ giữ các hình chữ nhật đủ lớn
            if cw >= self.min_card_width and ch >= self.min_card_height:
                # Crop card từ region
                card_img = region_img[y:y+ch, x:x+cw]
                
                # Bbox trong toạ độ ảnh gốc
                abs_y = y_offset + y
                
                cards.append({
                    'type': 'card',
                    'bbox': [x, abs_y, cw, ch],
                    'bbox_norm': [
                        x / img_width,
                        abs_y / img_height,
                        cw / img_width,
                        ch / img_height
                    ],
                    'image': card_img
                })
        
        return cards
    
    def visualize_components(self, image_path: str, output_path: str = None):
        """
        Vẽ bounding boxes lên ảnh để debug
        
        Args:
            image_path: Đường dẫn ảnh gốc
            output_path: Nơi lưu ảnh đã vẽ (nếu None thì show thôi)
        """
        img = cv2.imread(str(image_path))
        components = self.detect(image_path)
        
        # Màu cho từng loại component
        colors = {
            'header': (0, 255, 0),    # Green
            'footer': (0, 0, 255),    # Red
            'body': (255, 0, 0),      # Blue
            'card': (255, 255, 0)     # Cyan
        }
        
        for comp in components:
            x, y, w, h = comp['bbox']
            color = colors.get(comp['type'], (128, 128, 128))
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, comp['type'], (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Visualization saved to: {output_path}")
        else:
            cv2.imshow('UI Components', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test script
    detector = UIComponentDetector()
    
    # Test với một ảnh trong dataset
    test_image = Path("dataset/project_001_amazon/images/homepage.png")
    
    if test_image.exists():
        print(f"Testing with: {test_image}")
        components = detector.detect(str(test_image))
        
        print(f"\nDetected {len(components)} components:")
        for i, comp in enumerate(components, 1):
            print(f"  {i}. Type: {comp['type']}, BBox: {comp['bbox']}")
        
        # Visualize
        output = "test_component_detection.png"
        detector.visualize_components(str(test_image), output)
    else:
        print("Test image not found. Please update path.")
