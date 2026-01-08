"""
Component Utilities
Helper functions for building and managing component metadata
"""
import json
from typing import List, Dict


def build_components_metadata(components: List[Dict]) -> Dict:
    """
    Build component metadata structure from detected components
    
    Args:
        components: List of detected components with bbox, type, semantic_type
        
    Returns:
        Dictionary with structured metadata
    """
    if not components:
        return {
            "total_components": 0,
            "by_type": {},
            "layout_signature": "",
            "components": []
        }
    
    # Count by type and calculate average sizes
    by_type = {}
    for comp in components:
        semantic_type = comp.get('semantic_type', comp.get('type', 'unknown'))
        
        if semantic_type not in by_type:
            by_type[semantic_type] = {
                "count": 0,
                "total_size": 0.0
            }
        
        by_type[semantic_type]["count"] += 1
        
        # Calculate normalized area
        bbox_norm = comp.get('bbox_norm', [0, 0, 0, 0])
        if len(bbox_norm) == 4:
            area = bbox_norm[2] * bbox_norm[3]  # width * height
            by_type[semantic_type]["total_size"] += area
    
    # Calculate average sizes
    for type_name, stats in by_type.items():
        avg_size = stats["total_size"] / stats["count"] if stats["count"] > 0 else 0
        by_type[type_name]["avg_size"] = round(avg_size, 4)
        del stats["total_size"]
    
    # Generate layout signature (order of components from top to bottom)
    sorted_components = sorted(components, key=lambda c: c.get('bbox_norm', [0, 0, 0, 0])[1])
    layout_signature = "-".join([
        c.get('semantic_type', c.get('type', 'unknown')) 
        for c in sorted_components
    ])
    
    # Build component list (simplified for storage)
    component_list = []
    for comp in components:
        component_list.append({
            "type": comp.get('type', 'unknown'),
            "semantic_type": comp.get('semantic_type', comp.get('type', 'unknown')),
            "bbox": comp.get('bbox', [0, 0, 0, 0]),
            "bbox_norm": comp.get('bbox_norm', [0, 0, 0, 0]),
            "confidence": comp.get('confidence', 1.0)
        })
    
    return {
        "total_components": len(components),
        "by_type": by_type,
        "layout_signature": layout_signature,
        "components": component_list
    }


def metadata_to_json(metadata: Dict) -> str:
    """
    Convert metadata dictionary to JSON string for database storage
    
    Args:
        metadata: Component metadata dictionary
        
    Returns:
        JSON string
    """
    return json.dumps(metadata, ensure_ascii=False)


def json_to_metadata(json_str: str) -> Dict:
    """
    Parse JSON string back to metadata dictionary
    
    Args:
        json_str: JSON string from database
        
    Returns:
        Metadata dictionary
    """
    if not json_str:
        return {}
    return json.loads(json_str)


def get_component_types(metadata: Dict) -> List[str]:
    """
    Extract list of component types from metadata
    
    Args:
        metadata: Component metadata dictionary
        
    Returns:
        List of unique component types
    """
    return list(metadata.get('by_type', {}).keys())


def filter_components_by_type(components: List[Dict], component_type: str) -> List[Dict]:
    """
    Filter components by semantic type
    
    Args:
        components: List of components
        component_type: Type to filter (e.g., 'hero', 'header')
        
    Returns:
        Filtered list of components
    """
    return [
        c for c in components 
        if c.get('semantic_type', c.get('type')) == component_type
    ]


if __name__ == "__main__":
    # Test the utilities
    test_components = [
        {
            "type": "section",
            "semantic_type": "header",
            "bbox": [0, 0, 1920, 100],
            "bbox_norm": [0, 0, 1, 0.05],
            "confidence": 0.95
        },
        {
            "type": "section",
            "semantic_type": "hero",
            "bbox": [0, 100, 1920, 500],
            "bbox_norm": [0, 0.05, 1, 0.25],
            "confidence": 0.88
        }
    ]
    
    metadata = build_components_metadata(test_components)
    print("Metadata:", json.dumps(metadata, indent=2))
    
    json_str = metadata_to_json(metadata)
    print("\nJSON String:", json_str)
    
    restored = json_to_metadata(json_str)
    print("\nRestored:", restored)
