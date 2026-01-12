"""
Semantic UI Component Classifier
Phân loại các UI components thành semantic types (header, hero, footer, CTA, etc.)
Sử dụng hybrid approach: Rules + CLIP zero-shot
"""

from typing import List, Dict, Tuple
import numpy as np

class SemanticClassifier:
    """
    Classifier để phân loại UI components thành semantic types
    """
    
    def __init__(self, use_clip=False):
        """
        Args:
            use_clip: Enable CLIP zero-shot classification (slower but more accurate)
        """
        self.use_clip = use_clip
        self.clip_model = None
        self.clip_preprocess = None
        
        if use_clip:
            self._init_clip()
        
        # Define COMPLETE component taxonomy (25+ types)
        self.component_types = {
            # === LAYOUT STRUCTURE ===
            'header',           # Top navigation bar
            'navigation',       # Menu / Navigation bar
            'hero',            # Hero banner / Slider
            'banner',          # Promotional banner
            'promo_banner',    # Sale/promotion banner
            'breadcrumb',      # Breadcrumb trail
            'sidebar',         # Side navigation / column
            'main_content',    # Main content area
            'footer',          # Footer section
            'copyright',       # Copyright area
            
            # === CONTENT BLOCKS ===
            'section',         # Generic content section
            'article',         # Article / Blog post
            'widget',          # Small widget / box
            'gallery',         # Image gallery / grid
            'testimonials',    # Customer reviews
            'pricing_table',   # Pricing comparison table
            'faq',            # FAQ section
            
            # === INTERACTIVE ELEMENTS ===
            'cta_button',      # Call-to-action button
            'form',           # Any form (login, signup, contact, search)
            'login_form',     # Login form specifically
            'search_form',    # Search bar / form
            'input_field',    # Single input field
            
            # === SOCIAL & SUPPORT ===
            'social_links',   # Social media icons
            'chat_widget',    # Chat support button
            'modal',          # Popup / Modal
            'popup',          # Popup overlay
            
            # === SPECIALIZED ===
            'product_card',   # E-commerce product card
            'menu',           # Dropdown menu
            'logo',           # Company logo
        }
    
    def _init_clip(self):
        """Initialize CLIP model for zero-shot classification"""
        try:
            import clip
            import torch
            
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load CLIP model
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            
            # ENHANCED text prompts - ALIGNED WITH USER REQUEST
            self.clip_prompts = {
                # === STRUCTURE & NAVIGATION ===
                'header': "a website header navigation bar at the top with logo menu links and search box",
                'navigation': "a horizontal navigation menu bar with multiple menu links tabs and dropdown menus",
                'breadcrumb': "a breadcrumb navigation trail text showing page hierarchy path with arrows or slashes like Home > category > product",
                'sidebar': "a vertical sidebar column navigation menu on the left or right side with list of links categories filters",
                'footer': "a website footer section at the bottom with dark background columns links copyright contact info sitemap",
                'copyright': "small copyright text area at the very bottom of page showing year and company name rights reserved",
                
                # === CONTENT SECTIONS ===
                'hero_banner': "a large hero banner slider section at top of page with big headline text call to action button and background image",
                'promo_banner': "a promotional banner advertisement strip with sale discount offer text percentage off coupon code",
                'main_content': "the main body content area of the web page with paragraphs text articles",
                'section': "a distinct content section block with heading subheading and content body",
                'article': "an article post or blog entry with title author date and body text paragraphs",
                'gallery': "an image gallery grid layout with multiple photos thumbnails arranged in rows and columns",
                'testimonials': "customer testimonials reviews section with user profile pictures quotes stars rating and feedback text",
                'pricing_table': "a pricing plan comparison table with columns showing price tiers features list and subscribe buttons",
                'faq': "frequently asked questions FAQ section with list of questions and expandable answers",
                'features': "a features list section with icons and short text descriptions of product benefits",
                
                # === INTERACTIVE ELEMENTS ===
                'cta_button': "a prominent call to action button with bold text like buy now sign up get started",
                'form': "a form input area with text fields labels checkboxes and submit button",
                'login_form': "a login or sign in form with username email password fields and login button",
                'search_bar': "a search input field bar with magnifying glass icon and placeholder text",
                'widget': "a small standalone widget box with title and content or tools",
                'popup': "a popup modal overlay window with message content and close button",
                'chat_widget': "a live chat support button floating loop or icon in the bottom right corner",
                'social_links': "social media icons row with logos for facebook twitter instagram linkedin",
            }
            
            print(f"[INFO] CLIP model loaded on {self.device} for semantic classification")
            
        except ImportError:
            print("[WARN] CLIP not available. Install: pip install git+https://github.com/openai/CLIP.git")
            self.use_clip = False
    
    def classify(self, component: Dict, img_shape: Tuple[int, int], all_components: List[Dict] = None) -> str:
        """
        Hybrid Classification: Rules -> CLIP -> Validation
        """
        # 0. Structural Constraints (Hard Filters)
        # Prevents "Sidebar" detection on a square product photo
        bbox = component['bbox']
        w, h = bbox[2], bbox[3]
        aspect_ratio = w / (h + 1e-6)
        
        # Valid candidates based on shape
        valid_types = set(self.clip_prompts.keys())
        
        # Sidebar MUST be tall (AR < 0.6)
        if aspect_ratio >= 0.6:
            valid_types.discard('sidebar')
            
        # Header/Nav MUST be wide (AR > 2.0)
        if aspect_ratio <= 2.0:
            valid_types.discard('header')
            valid_types.discard('navigation')
            
        # 1. CLIP Classification
        if self.use_clip:
            clip_type, confidence = self._classify_by_clip(component, valid_types)
            
            # Post-check: Is it actually a UI element?
            if clip_type in ['product_photo', 'background_graphic']:
                return 'element' # Treat as generic content
                
            if confidence > 0.22:
                return clip_type
        
        # Priority 2: Strong structural rules (very obvious only)
        strong_type = self._classify_strong_rules(component, img_shape)
        if strong_type:
            return strong_type
        
        # Priority 3: Weak rules (fallback)
        weak_type = self._classify_weak_rules(component, img_shape)
        if weak_type:
            return weak_type
        
        # Final fallback
        return self._fallback_classification(component, img_shape)
    
    def _classify_strong_rules(self, component: Dict, img_shape: Tuple[int, int]) -> str:
        """
        VERY STRICT structural rules - Only absolute obvious cases
        Most components should go through CLIP instead
        """
        bbox = component['bbox']
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        img_h, img_w = img_shape[:2]
        
        # Normalized metrics
        y_ratio = y / img_h
        x_ratio = x / img_w
        width_ratio = w / img_w
        height_ratio = h / img_h
        area_ratio = (w * h) / (img_w * img_h)
        
        # ONLY ABSOLUTE OBVIOUS CASES (strictest thresholds)
        
        # Header: Top 8% (very strict), FULL width (>90%), absolute y < 100px
        if y_ratio < 0.08 and width_ratio > 0.90 and y < 100:
            return 'header'
        
        # Footer: DISABLED for viewport screenshots
        # Viewport screenshots usually don't have footer
        # Let CLIP handle footer classification based on visual content (copyright, links, etc.)
        # if y_ratio > 0.95 and width_ratio > 0.90:
        #     return 'footer'
        
        # Chat Widget: Extreme bottom-right corner, tiny
        if x_ratio > 0.90 and y_ratio > 0.85 and area_ratio < 0.02:
            return 'chat_widget'
        
        # ALL OTHER CASES → Go to CLIP
        return None
    
    def _classify_weak_rules(self, component: Dict, img_shape: Tuple[int, int]) -> str:
        """
        Weak heuristic rules - Lower confidence, more assumptions
        Used when strong rules and CLIP don't match
        """
        bbox = component['bbox']
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        img_h, img_w = img_shape[:2]
        
        y_ratio = y / img_h
        x_ratio = x / img_w
        width_ratio = w / img_w
        height_ratio = h / img_h
        area_ratio = (w * h) / (img_w * img_h)
        aspect_ratio = w / (h + 1e-6)
        
        # Possible header (wider threshold)
        if y_ratio < 0.20 and width_ratio > 0.6:
            return 'header'
        
        # Possible breadcrumb
        if y_ratio < 0.25 and height_ratio < 0.05 and width_ratio > 0.3 and aspect_ratio > 5:
            return 'breadcrumb'
        
        # Possible sidebar (relaxed)
        if width_ratio < 0.30 and height_ratio > 0.35:
            return 'sidebar'
        
        # Small-medium components in upper-middle area might be CTA
        if 0.15 < y_ratio < 0.5 and 0.01 < area_ratio < 0.08:
            return 'cta_button'
        
        # Thin horizontal elements might be forms
        if height_ratio < 0.08 and width_ratio > 0.25 and aspect_ratio > 4:
            return 'form'
        
        return None
    
    def _classify_by_clip(self, component: Dict, allowed_types: set = None) -> Tuple[str, float]:
        """
        CLIP zero-shot classification with restricted candidates
        
        Returns:
            (component_type, confidence)
        """
        if not self.clip_model:
            return 'unknown', 0.0
        
        try:
            import torch
            import clip
            from PIL import Image
            import cv2
            
            # Convert component image to PIL
            img_array = component.get('image')
            if img_array is None or img_array.size == 0:
                return 'unknown', 0.0
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            
            # Filter prompts based on allowed_types
            active_prompts = {k: v for k, v in self.clip_prompts.items() 
                             if allowed_types is None or k in allowed_types}
            
            if not active_prompts:
                return 'unknown', 0.0
                
            labels = list(active_prompts.keys())
            texts = list(active_prompts.values())
            
            # Preprocess
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Tokenize prompts
            text_inputs = clip.tokenize(texts).to(self.device)
            
            # Calculate similarity
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Cosine similarity with SOFTMAX
                similarity = (image_features @ text_features.T).softmax(dim=-1)
            
            # Get best match
            best_idx = similarity.argmax().item()
            confidence = similarity[0, best_idx].item()
            
            best_label = labels[best_idx]
            
            return best_label, confidence
            
        except Exception as e:
            print(f"[WARN] CLIP classification failed: {e}")
            return 'unknown', 0.0
    
    def _fallback_classification(self, component: Dict, img_shape: Tuple[int, int]) -> str:
        """
        Fallback classification when no rule/CLIP match
        """
        bbox = component['bbox']
        area_ratio = (bbox[2] * bbox[3]) / (img_shape[0] * img_shape[1])
        
        # Large components -> section
        if area_ratio > 0.15:
            return 'section'
        
        # Medium -> widget
        if area_ratio > 0.05:
            return 'widget'
        
        # Small -> element (generic)
        return 'element'
    
    def classify_all(self, components: List[Dict], img_shape: Tuple[int, int]) -> List[Dict]:
        """
        Classify all components and add 'semantic_type' field
        """
        for comp in components:
            semantic_type = self.classify(comp, img_shape, components)
            comp['semantic_type'] = semantic_type
        
        # Post-processing: Context-aware refinement
        components = self._refine_with_context(components, img_shape)
        
        return components
    
    def _refine_with_context(self, components: List[Dict], img_shape: Tuple[int, int]) -> List[Dict]:
        """
        Use spatial relationships and patterns to refine classification
        """
        # Detect gallery pattern (grid of similar-sized images)
        gallery_candidates = self._detect_gallery_pattern(components, img_shape)
        for comp_idx in gallery_candidates:
            components[comp_idx]['semantic_type'] = 'gallery'
        
        # Detect form pattern (group of input-like elements)
        form_groups = self._detect_form_pattern(components, img_shape)
        for comp_idx in form_groups:
            components[comp_idx]['semantic_type'] = 'form'
        
        return components
    
    def _detect_gallery_pattern(self, components: List[Dict], img_shape: Tuple[int, int]) -> List[int]:
        """
        Detect grid patterns that indicate gallery
        Returns indices of components that are part of gallery
        """
        gallery_indices = []
        
        # Find clusters of similar-sized components
        # (Simple heuristic: if 4+ components with similar size are arranged in grid)
        
        return gallery_indices
    
    def _detect_form_pattern(self, components: List[Dict], img_shape: Tuple[int, int]) -> List[int]:
        """
        Detect groups of thin horizontal boxes (likely input fields)
        """
        form_indices = []
        
        # Find thin horizontal components grouped vertically
        # Typical form: multiple boxes with height < 5% and stacked vertically
        
        return form_indices
