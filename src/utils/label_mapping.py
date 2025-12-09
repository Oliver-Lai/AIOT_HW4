"""
Label Mapping Generator for EMNIST ByClass Dataset

Generates mapping between class indices (0-61) and characters
(digits 0-9, uppercase A-Z, lowercase a-z)
"""

import json
from pathlib import Path


def create_label_mapping():
    """
    Create mapping from EMNIST ByClass indices to characters.
    
    EMNIST ByClass has 62 classes:
    - Indices 0-9: Digits '0'-'9'
    - Indices 10-35: Uppercase 'A'-'Z'
    - Indices 36-61: Lowercase 'a'-'z'
    
    Returns:
        dict: Mapping from index to character
    """
    mapping = {}
    
    # Digits 0-9 (indices 0-9)
    for i in range(10):
        mapping[i] = str(i)
    
    # Uppercase A-Z (indices 10-35)
    for i in range(26):
        mapping[10 + i] = chr(ord('A') + i)
    
    # Lowercase a-z (indices 36-61)
    for i in range(26):
        mapping[36 + i] = chr(ord('a') + i)
    
    return mapping


def create_reverse_mapping(mapping):
    """
    Create reverse mapping from character to index.
    
    Args:
        mapping: Dictionary mapping index to character
        
    Returns:
        dict: Mapping from character to index
    """
    return {v: k for k, v in mapping.items()}


def save_label_mapping(output_path="models/label_mapping.json"):
    """
    Generate and save label mapping to JSON file.
    
    Args:
        output_path: Path to save the JSON file
        
    Returns:
        dict: The generated mapping
    """
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate mapping
    mapping = create_label_mapping()
    
    # Save to JSON (convert int keys to strings for JSON compatibility)
    json_mapping = {str(k): v for k, v in mapping.items()}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Label mapping saved to {output_file}")
    print(f"  Total classes: {len(mapping)}")
    print(f"  Digits (0-9): indices 0-9")
    print(f"  Uppercase (A-Z): indices 10-35")
    print(f"  Lowercase (a-z): indices 36-61")
    
    return mapping


def load_label_mapping(mapping_path="models/label_mapping.json"):
    """
    Load label mapping from JSON file.
    
    Args:
        mapping_path: Path to the JSON file
        
    Returns:
        dict: Mapping from index (int) to character (str)
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        json_mapping = json.load(f)
    
    # Convert string keys back to integers
    mapping = {int(k): v for k, v in json_mapping.items()}
    
    return mapping


def get_character(index, mapping=None):
    """
    Get character for a given class index.
    
    Args:
        index: Class index (0-61)
        mapping: Optional pre-loaded mapping dictionary
        
    Returns:
        str: Corresponding character
    """
    if mapping is None:
        mapping = create_label_mapping()
    
    return mapping.get(index, '?')


def get_index(character, reverse_mapping=None):
    """
    Get class index for a given character.
    
    Args:
        character: Character ('0'-'9', 'A'-'Z', 'a'-'z')
        reverse_mapping: Optional pre-loaded reverse mapping
        
    Returns:
        int: Corresponding class index, or None if not found
    """
    if reverse_mapping is None:
        mapping = create_label_mapping()
        reverse_mapping = create_reverse_mapping(mapping)
    
    return reverse_mapping.get(character)


def print_mapping_samples():
    """Print sample mappings for verification."""
    mapping = create_label_mapping()
    reverse_mapping = create_reverse_mapping(mapping)
    
    print("\n" + "="*60)
    print("EMNIST ByClass Label Mapping")
    print("="*60)
    
    print("\nSample Digits:")
    for i in range(10):
        print(f"  Index {i:2d} → '{mapping[i]}'")
    
    print("\nSample Uppercase Letters:")
    for i in range(10, 20):
        print(f"  Index {i:2d} → '{mapping[i]}'")
    
    print("\nSample Lowercase Letters:")
    for i in range(36, 46):
        print(f"  Index {i:2d} → '{mapping[i]}'")
    
    print("\nReverse Lookup Examples:")
    test_chars = ['5', 'A', 'Z', 'a', 'z']
    for char in test_chars:
        idx = reverse_mapping[char]
        print(f"  '{char}' → Index {idx}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("EMNIST Label Mapping Generator")
    print("="*60)
    
    # Save mapping
    mapping = save_label_mapping()
    
    # Verify it can be loaded
    print("\nVerifying saved mapping...")
    loaded_mapping = load_label_mapping()
    assert loaded_mapping == mapping, "Loaded mapping doesn't match!"
    print("✓ Mapping verified successfully")
    
    # Print samples
    print_mapping_samples()
    
    print("\n✅ Label mapping creation complete!")
