import torch
import clip
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple

class EmbeddingService:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EmbeddingService] Loading CLIP on {self.device}...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        
        # Check OCR availability
        self.ocr_available = False
        try:
            import pytesseract
            # Test if tesseract is strictly installed
            # pytesseract.get_tesseract_version()
            self.ocr_available = True
        except ImportError:
            print("[EmbeddingService] Pytesseract not found. OCR disabled.")
        except Exception:
            self.ocr_available = False

    def _preprocess_with_padding(self, image: np.ndarray) -> torch.Tensor:
        """
        Pad image to square (black padding) then resize to 224x224 to preserve aspect ratio.
        Normalization matches CLIP expected mean/std.
        """
        target_size = 224
        h, w = image.shape[:2]
        
        # 1. Pad to square
        dim = max(h, w)
        pad_h = (dim - h) // 2
        pad_w = (dim - w) // 2
        
        # Create black canvas
        padded = np.zeros((dim, dim, 3), dtype=np.uint8)
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
        
        # 2. Resize to 224x224
        resized = cv2.resize(padded, (target_size, target_size))
        
        # 3. Convert to Tensor & Normalize
        # CLIP expects RGB, 0-1 range, then normalized
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        
        # Standard CLIP mean/std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(3, 1, 1)
        
        img_tensor = (img_tensor.to(self.device) - mean) / std
        return img_tensor.unsqueeze(0)

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Generate CLIP embedding for a CV2 BGR image.
        Returns: 1D numpy array (normalized)
        """
        if image is None or image.size == 0:
            return None
            
        try:
            # Use custom padding preprocess instead of crop-heavy default
            img_tensor = self._preprocess_with_padding(image)
            
            with torch.no_grad():
                emb = self.clip_model.encode_image(img_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                
            return emb.cpu().numpy().flatten()
        except Exception as e:
            print(f"[EmbeddingService] Embedding error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate CLIP embedding for a text string.
        Returns: 1D numpy array (normalized)
        """
        if not text:
            return None
            
        try:
            # Prepare text
            text_tokens = clip.tokenize([text[:77]]).to(self.device) # CLIP limit 77 tokens
            
            with torch.no_grad():
                emb = self.clip_model.encode_text(text_tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                
            return emb.cpu().numpy().flatten()
        except Exception as e:
            print(f"[EmbeddingService] Text embedding error: {e}")
            return None

    def get_ocr_text(self, image: np.ndarray, limit: int = 500) -> str:
        """
        Extract standardized OCR text from CV2 BGR image.
        """
        if not self.ocr_available or image is None or image.size == 0:
            return ""
            
        try:
            import pytesseract
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Simple config for block text
            config = r'--psm 6' 
            
            text = pytesseract.image_to_string(pil_img, lang='eng', config=config)
            
            # Standardization
            text = text.lower().strip()
            # Normalize whitespace (replace newlines/tabs with single space)
            text = ' '.join(text.split())
            
            # Limit length
            return text[:limit]
            
        except Exception as e:
            print(f"[EmbeddingService] OCR error: {e}")
            return ""

    def process_component(self, image: np.ndarray, bbox=None) -> Dict:
        """
        Convenience wrapper to get both embedding and text.
        """
        return {
            'embedding': self.get_embedding(image),
            'ocr_text': self.get_ocr_text(image)
        }
