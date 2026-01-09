"""
Component Embedder - Generate CLIP embeddings for UI components
"""

import cv2
import numpy as np
import torch
from pathlib import Path

import clip
from PIL import Image


class ComponentEmbedder:
    """
    Generate CLIP embeddings for individual UI components
    """
    
    def __init__(self, model_name='ViT-B/32', device=None):
        """
        Initialize CLIP model for component embedding
        
        Args:
            model_name: CLIP model variant ('ViT-B/32' produces 512-dim vectors)
            device: torch device, auto-detects if None
        """
        print(f"[ComponentEmbedder] Loading CLIP model: {model_name}...")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        print(f"[ComponentEmbedder] ✓ CLIP loaded on {device}")
    
    def embed_component(self, image, bbox):
        """
        Generate embedding for a single component
        
        Args:
            image: Full image (numpy array, BGR format from cv2)
            bbox: Bounding box [x, y, w, h]
            
        Returns:
            numpy array of shape (512,) - normalized embedding
        """
        x, y, w, h = bbox
        
        # Crop component
        component_img = image[y:y+h, x:x+w]
        
        # Convert BGR to RGB
        component_rgb = cv2.cvtColor(component_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        pil_img = Image.fromarray(component_rgb)
        
        # Preprocess and encode
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
        
        return embedding.cpu().numpy().flatten()
    
    def embed_components(self, image_path, components):
        """
        Add 'embedding' field to each component in the list
        
        Args:
            image_path: Path to full image
            components: List of component dicts with 'bbox' field
            
        Returns:
            Updated components list with 'embedding' added
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Generate embeddings
        for comp in components:
            bbox = comp.get('bbox')
            if bbox:
                embedding = self.embed_component(img, bbox)
                comp['embedding'] = embedding.tolist()  # Convert to list for JSON
        
        return components
    
    def batch_embed(self, image, bboxes, batch_size=16):
        """
        Batch process multiple components for better performance
        
        Args:
            image: Full image (numpy array)
            bboxes: List of bounding boxes
            batch_size: Number of components to process at once
            
        Returns:
            List of embeddings (numpy arrays)
        """
        embeddings = []
        
        for i in range(0, len(bboxes), batch_size):
            batch_boxes = bboxes[i:i+batch_size]
            batch_images = []
            
            # Prepare batch
            for bbox in batch_boxes:
                x, y, w, h = bbox
                component = image[y:y+h, x:x+w]
                component_rgb = cv2.cvtColor(component, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(component_rgb)
                preprocessed = self.preprocess(pil_img)
                batch_images.append(preprocessed)
            
            # Stack and encode batch
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                batch_embeddings = self.model.encode_image(batch_tensor)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            
            embeddings.extend(batch_embeddings.cpu().numpy())
        
        return embeddings


if __name__ == "__main__":
    # Quick test
    print("Testing ComponentEmbedder...")
    
    embedder = ComponentEmbedder()
    
    # Test with dummy data
    test_img = np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8)
    test_bbox = [0, 0, 300, 100]
    
    embedding = embedder.embed_component(test_img, test_bbox)
    
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding norm: {np.linalg.norm(embedding):.4f}")
    print("✓ ComponentEmbedder working!")
