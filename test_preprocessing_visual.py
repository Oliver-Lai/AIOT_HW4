"""
Visual test for canvas preprocessing to verify images are processed correctly.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_test_canvas():
    """Create a simulated canvas with a drawn character."""
    # Create white background (280x280 RGBA)
    canvas = np.ones((280, 280, 4), dtype=np.uint8) * 255
    
    # Draw a simple "A" shape in black
    # Vertical left line
    canvas[80:200, 100:110, 0:3] = 0
    # Vertical right line
    canvas[80:200, 170:180, 0:3] = 0
    # Horizontal middle line
    canvas[130:140, 100:180, 0:3] = 0
    # Top horizontal line
    canvas[80:90, 110:170, 0:3] = 0
    
    return canvas

def preprocess_canvas(canvas_data):
    """Preprocess canvas image (same logic as app.py)."""
    # Convert RGBA to grayscale
    img_rgb = canvas_data[:, :, :3]
    img_gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Check if empty
    if img_gray.min() > 250:
        return None, None
    
    # Resize to 28x28
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert colors (black-on-white -> white-on-black)
    img_inverted = 255 - img_resized
    
    # Normalize to [0, 1]
    img_normalized = img_inverted.astype(np.float32) / 255.0
    
    # Reshape for model
    img_final = img_normalized.reshape(1, 28, 28, 1)
    
    return img_gray, img_final

def main():
    """Run visual test."""
    print("Creating test canvas...")
    canvas = create_test_canvas()
    
    print("Preprocessing canvas...")
    img_gray, img_processed = preprocess_canvas(canvas)
    
    if img_processed is None:
        print("❌ Canvas was detected as empty!")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original canvas (RGB view)
    axes[0].imshow(canvas[:, :, :3])
    axes[0].set_title("Original Canvas\n(User draws black on white)", fontsize=12)
    axes[0].axis('off')
    
    # Grayscale version
    axes[1].imshow(img_gray, cmap='gray')
    axes[1].set_title("Grayscale\n(280x280)", fontsize=12)
    axes[1].axis('off')
    
    # Resized and inverted (28x28)
    axes[2].imshow(img_processed[0, :, :, 0], cmap='gray')
    axes[2].set_title("Preprocessed for Model\n(28x28, white-on-black)", fontsize=12)
    axes[2].axis('off')
    
    # Display with interpolation for clarity
    axes[3].imshow(img_processed[0, :, :, 0], cmap='gray', interpolation='nearest')
    axes[3].set_title("Preprocessed (Zoomed)\n(Nearest neighbor)", fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_preprocessing_output.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to: test_preprocessing_output.png")
    
    # Print statistics
    print("\nPreprocessing Statistics:")
    print(f"  Input shape: {canvas.shape}")
    print(f"  Output shape: {img_processed.shape}")
    print(f"  Value range: [{img_processed.min():.3f}, {img_processed.max():.3f}]")
    print(f"  Non-zero pixels: {(img_processed > 0.1).sum()} / 784")
    print(f"  Mean intensity: {img_processed.mean():.3f}")
    print(f"  Max intensity: {img_processed.max():.3f}")
    
    # Verify it's not all black or all white
    if img_processed.max() > 0.5 and img_processed.min() < 0.5:
        print("\n✅ Image preprocessing is working correctly!")
        print("   The drawn character is visible in the preprocessed image.")
    else:
        print("\n⚠️ Warning: Image might be too uniform")

if __name__ == "__main__":
    main()
