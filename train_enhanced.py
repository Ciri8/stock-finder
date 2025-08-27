"""Enhanced training script with data augmentation and better preprocessing"""

import numpy as np
import torch
from train_with_real_data import collect_sp500_data, create_labeled_dataset, train_on_real_data
import random

def augment_chart_data(image, label):
    """Apply data augmentation to chart images"""
    augmented = []
    
    # Original
    augmented.append((image, label))
    
    # Add noise
    noise = np.random.normal(0, 0.01, image.shape)
    augmented.append((image + noise, label))
    
    # Scale variation
    scale = random.uniform(0.95, 1.05)
    scaled = image * scale
    augmented.append((scaled, label))
    
    # Small time shift (if not critical for pattern)
    if label != 0:  # Don't shift for no_pattern
        shift = random.randint(-2, 2)
        if shift != 0:
            shifted = np.roll(image, shift, axis=-1)
            augmented.append((shifted, label))
    
    return augmented

def enhance_training():
    """Run enhanced training with all improvements"""
    
    print("="*70)
    print(" ENHANCED PATTERN RECOGNITION TRAINING")
    print("="*70)
    
    # Collect more data
    print("\nCollecting S&P 500 data (this will take a few minutes)...")
    stock_data = collect_sp500_data(limit=300)  # Even more stocks
    
    # Create dataset with aggressive sampling
    print("\nCreating dataset with augmentation...")
    images, labels, metadata = create_labeled_dataset(
        stock_data, 
        window_size=60, 
        stride=5  # Very small stride for maximum samples
    )
    
    # Apply data augmentation
    augmented_images = []
    augmented_labels = []
    augmented_metadata = []
    
    for img, lbl, meta in zip(images, labels, metadata):
        aug_samples = augment_chart_data(img, lbl)
        for aug_img, aug_lbl in aug_samples:
            augmented_images.append(aug_img)
            augmented_labels.append(aug_lbl)
            augmented_metadata.append(meta)
    
    print(f"\nDataset size after augmentation: {len(augmented_images)}")
    
    # Train with optimal parameters
    recognizer, accuracy = train_on_real_data(
        augmented_images, 
        augmented_labels, 
        augmented_metadata,
        epochs=100  # Many epochs with early stopping
    )
    
    print(f"\nFinal accuracy: {accuracy:.2f}%")
    print("\nModel saved to: data/models/real_data_pattern_model.pth")
    
    return recognizer, accuracy

if __name__ == "__main__":
    enhance_training()