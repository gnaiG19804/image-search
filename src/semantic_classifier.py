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
            
            # ENHANCED text prompts - VERY DETAILED for better accuracy
            self.clip_prompts = {
                # Layout Structure
                'header': "a website header navigation bar at the top with logo menu links and search box on white or colored background",
                'navigation': "a horizontal navigation menu bar with multiple menu links tabs and dropdown menus",
                'hero': "a large hero banner section with big bold headline text call to action button and attractive background image or gradient at the top of page",
                'banner': "a promotional banner or advertising banner with colorful design product images sale text discount percentage bright colors",
                'promo_banner': "a sale promotion banner with bold discount percentages special offer text limited time sale flash sale coupon code on vibrant red orange yellow background",
                'breadcrumb': "a breadcrumb navigation trail showing page hierarchy with home arrow links separated by slashes or arrows",
                'sidebar': "a sidebar column or vertical navigation menu with links widgets on the left or right side of page",
                'main_content': "the main content area body section with paragraphs text images and content",
                'footer': "a website footer section at bottom with dark background multiple columns sitemap links copyright text contact info social icons",
                'copyright': "a copyright text area with small text showing year company name and rights reserved",
                
                # Content Blocks
                'section': "a content section block with heading subheading and body text paragraphs",
                'article': "an article or blog post with title author date paragraphs images and formatted text content",
                'widget': "a small widget box component with border icon title and compact content",
                'gallery': "a photo gallery grid layout showing multiple images arranged in rows and columns with thumbnails",
                'testimonials': "customer testimonials reviews section with user profile photos quotes star ratings and customer feedback",
                'pricing_table': "a pricing plan comparison table with multiple pricing tiers columns showing price features and buy buttons",
                'faq': "frequently asked questions FAQ section with collapsible expandable questions and answers accordion style",
                
                # Interactive Elements
                'cta_button': "a bright colorful call to action button with bold text like buy now sign up get started subscribe on vibrant background",
                'form': "a form interface with multiple input fields textboxes labels and submit button for data entry",
                'login_form': "a login signin form card with email username input password field remember me checkbox and login button",
                'search_form': "a search bar input box with magnifying glass icon search button and placeholder text",
                'input_field': "a single text input box field for typing text with border and placeholder text",
                
                # Social & Support  
                'social_links': "social media icons links for facebook twitter instagram youtube linkedin pinterest",
                'chat_widget': "a live chat support button widget icon in bottom right corner with message bubble icon",
                'modal': "a popup modal dialog box overlay window with close button title content on top of dimmed background",
                'popup': "a popup overlay window notification banner with message close button and call to action",
                
                # Specialized
                'product_card': "an ecommerce product card showing product image name price rating stars add to cart button",
                'menu': "a dropdown menu panel with list of menu items links and submenu options",
                'logo': "a company brand logo icon or logotype image at top left corner",
            }
            
            print(f"[INFO] CLIP model loaded on {self.device} for semantic classification")
            
        except ImportError:
            print("[WARN] CLIP not available. Install: pip install git+https://github.com/openai/CLIP.git")
            self.use_clip = False
    
    def classify(self, component: Dict, img_shape: Tuple[int, int], all_components: List[Dict] = None) -> str:
        """
        CLIP-FIRST Hybrid Classification (Visual Content Priority)
        
        Priority 1: CLIP zero-shot (visual matching) - PRIMARY METHOD
        Priority 2: Strong structural rules (only very obvious cases)
        Priority 3: Weak rules (last resort fallback)
        
        Args:
            component: Component dict with bbox, image, etc.
            img_shape: (height, width) of original image
            all_components: List of all components for context analysis
            
        Returns:
            Component type string
        """
        # Priority 1: CLIP FIRST (visual content matching)
        if self.use_clip:
            clip_type, confidence = self._classify_by_clip(component)
            # Balanced threshold for good precision/recall
            if confidence > 0.25:  # Working threshold from POC
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
    
    def _classify_by_clip(self, component: Dict) -> Tuple[str, float]:
        """
        CLIP zero-shot classification
        
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
            
            # Preprocess
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Tokenize prompts
            text_inputs = clip.tokenize(list(self.clip_prompts.values())).to(self.device)
            
            # Calculate similarity
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Cosine similarity with SOFTMAX (proper probability distribution)
                similarity = (image_features @ text_features.T).softmax(dim=-1)
            
            # Get best match
            best_idx = similarity.argmax().item()
            confidence = similarity[0, best_idx].item()
            
            component_type = list(self.clip_prompts.keys())[best_idx]
            
            return component_type, confidence
            
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
