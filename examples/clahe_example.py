#!/usr/bin/env python3
"""
Example usage of DALI CLAHE operator.
This demonstrates how to use CLAHE (Contrast Limited Adaptive Histogram Equalization)
in a DALI pipeline for image preprocessing.
"""

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np


def create_clahe_pipeline(batch_size=4, num_threads=2, device_id=0, image_dir=None):
    """
    Create a DALI pipeline with CLAHE operator.

    Args:
        batch_size: Number of images per batch
        num_threads: Number of worker threads
        device_id: GPU device ID
        image_dir: Directory containing images (if None, uses synthetic data)

    Returns:
        DALI pipeline with CLAHE preprocessing
    """

    @dali.pipeline_def(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id
    )
    def clahe_preprocessing_pipeline():
        if image_dir:
            # Read images from directory
            images, labels = fn.readers.file(file_root=image_dir, random_shuffle=True)
            images = fn.decoders.image(images, device="mixed")  # Decode on GPU

            # Resize to consistent size
            images = fn.resize(images, size=[256, 256])
        else:
            # Create synthetic test images with varying contrast
            # This simulates real-world scenarios where CLAHE is beneficial
            images = fn.random.uniform(
                range=(0, 255), shape=(256, 256, 3), dtype=types.UINT8
            )

            # Add some contrast variation to make CLAHE effect visible
            images = fn.cast(images, dtype=types.FLOAT32)

            # Simulate low contrast regions
            contrast_factor = fn.random.uniform(range=(0.3, 0.8))
            images = images * contrast_factor

            # Add brightness variation
            brightness_offset = fn.random.uniform(range=(-30, 30))
            images = images + brightness_offset

            # Clamp to valid range and convert back to uint8
            images = fn.clamp(images, 0, 255)
            images = fn.cast(images, dtype=types.UINT8)

        # Apply CLAHE for adaptive histogram equalization
        # Parameters:
        # - tiles_x, tiles_y: Number of tiles for local processing
        # - clip_limit: Threshold to limit contrast amplification (prevents noise)
        # - luma_only: For RGB, apply only to luminance channel (preserves color)
        clahe_images = fn.clahe(
            images,
            tiles_x=8,  # 8x8 grid of tiles
            tiles_y=8,
            clip_limit=2.0,  # Moderate clipping
            luma_only=True,
        )  # RGB: process luminance only

        return images, clahe_images

    return clahe_preprocessing_pipeline()


def demonstrate_clahe_parameters():
    """Demonstrate different CLAHE parameter settings."""

    @dali.pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def parameter_demo_pipeline():
        # Create a test image with poor contrast
        base_image = fn.random.uniform(
            range=(80, 120), shape=(256, 256, 1), dtype=types.UINT8
        )

        # Different CLAHE configurations
        clahe_default = fn.clahe(base_image, tiles_x=8, tiles_y=8, clip_limit=2.0)
        clahe_aggressive = fn.clahe(base_image, tiles_x=16, tiles_y=16, clip_limit=4.0)
        clahe_gentle = fn.clahe(base_image, tiles_x=4, tiles_y=4, clip_limit=1.0)

        return base_image, clahe_default, clahe_aggressive, clahe_gentle

    return parameter_demo_pipeline()


def main():
    """Main function demonstrating CLAHE usage."""

    print("DALI CLAHE Operator Example")
    print("=" * 40)

    try:
        # Create and build pipeline
        print("Creating CLAHE pipeline...")
        pipe = create_clahe_pipeline(batch_size=2, num_threads=1, device_id=0)
        pipe.build()
        print("✓ Pipeline built successfully")

        # Run pipeline
        print("\nRunning pipeline...")
        outputs = pipe.run()
        original_images, clahe_images = outputs

        # Move to CPU for analysis
        original_batch = original_images.as_cpu()
        clahe_batch = clahe_images.as_cpu()

        print(f"✓ Processed {len(original_batch)} images")

        # Analyze results
        for i in range(len(original_batch)):
            original = np.array(original_batch[i])
            enhanced = np.array(clahe_batch[i])

            print(f"\nImage {i + 1}:")
            print(
                f"  Original  - Shape: {original.shape}, Range: [{original.min()}, {original.max()}]"
            )
            print(
                f"  Enhanced  - Shape: {enhanced.shape}, Range: [{enhanced.min()}, {enhanced.max()}]"
            )

            # Calculate contrast metrics
            orig_std = np.std(original)
            enhanced_std = np.std(enhanced)
            contrast_improvement = enhanced_std / orig_std if orig_std > 0 else 1.0

            print(f"  Contrast improvement: {contrast_improvement:.2f}x")

        print("\n🎉 CLAHE pipeline executed successfully!")

        # Demonstrate parameter variations
        print("\nTesting different CLAHE parameters...")
        param_pipe = demonstrate_clahe_parameters()
        param_pipe.build()

        param_outputs = param_pipe.run()
        base, default, aggressive, gentle = param_outputs

        base_img = np.array(base.as_cpu()[0])
        default_img = np.array(default.as_cpu()[0])
        aggressive_img = np.array(aggressive.as_cpu()[0])
        gentle_img = np.array(gentle.as_cpu()[0])

        print(f"Base image std: {np.std(base_img):.2f}")
        print(f"Default CLAHE std: {np.std(default_img):.2f}")
        print(f"Aggressive CLAHE std: {np.std(aggressive_img):.2f}")
        print(f"Gentle CLAHE std: {np.std(gentle_img):.2f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure DALI is built with CLAHE operator support.")
        return False

    return True


if __name__ == "__main__":
    success = main()

    if success:
        print("\n" + "=" * 60)
        print("USAGE SUMMARY:")
        print("=" * 60)
        print("The CLAHE operator is now available in DALI!")
        print()
        print("Basic usage:")
        print("  enhanced = fn.clahe(images, tiles_x=8, tiles_y=8, clip_limit=2.0)")
        print()
        print("Parameters:")
        print("  - tiles_x, tiles_y: Grid size for local processing (4-16 typical)")
        print("  - clip_limit: Contrast amplification limit (1.0-4.0 typical)")
        print("  - luma_only: For RGB, process only luminance (default: True)")
        print("  - bins: Histogram bins (default: 256)")
        print()
        print("Use cases:")
        print("  - Medical imaging preprocessing")
        print("  - Low-light image enhancement")
        print("  - Improving contrast in shadowed regions")
        print("  - Satellite/aerial image processing")
