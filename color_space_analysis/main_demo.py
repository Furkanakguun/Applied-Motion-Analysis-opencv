"""
Main Demo Script for Color Space Analysis
Author: fakgun
Description: Comprehensive demo showcasing all color analysis techniques
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from hsv_segmentation import HSVSegmentation
from watershed_segmentation import WatershedSegmentation, create_test_image
from color_tracking import ColorTracker, demo_color_tracking
from advanced_color_analysis import AdvancedColorAnalysis

def create_comprehensive_test_image():
    """
    Create a comprehensive test image with various color objects
    """
    # Create base image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 50
    
    # Add colored circles (for HSV segmentation)
    colors = [
        ((0, 0, 255), (100, 100)),      # Red
        ((0, 255, 0), (200, 100)),      # Green
        ((255, 0, 0), (300, 100)),      # Blue
        ((0, 255, 255), (400, 100)),    # Yellow
        ((0, 165, 255), (500, 100)),    # Orange
    ]
    
    for color, pos in colors:
        cv2.circle(image, pos, 40, color, -1)
    
    # Add touching objects (for watershed)
    cv2.circle(image, (150, 250), 30, (255, 100, 100), -1)
    cv2.circle(image, (180, 250), 30, (100, 255, 100), -1)
    cv2.circle(image, (210, 250), 30, (100, 100, 255), -1)
    
    # Add rectangular objects
    cv2.rectangle(image, (300, 220), (350, 280), (255, 255, 100), -1)
    cv2.rectangle(image, (330, 220), (380, 280), (255, 100, 255), -1)
    
    # Add some noise and texture
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    # Add gradient background
    for i in range(100):
        color = (i * 2, 128, 255 - i * 2)
        cv2.line(image, (0, 350 + i), (600, 350 + i), color, 1)
    
    return image

def demo_hsv_segmentation():
    """
    Demo HSV color segmentation
    """
    print("=" * 50)
    print("HSV COLOR SEGMENTATION DEMO")
    print("=" * 50)
    
    # Create test image
    image = create_comprehensive_test_image()
    
    # Initialize segmenter
    segmenter = HSVSegmentation()
    
    # Test different colors
    colors = ["red", "green", "blue", "yellow", "orange"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Test Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Segment each color
    for i, color in enumerate(colors):
        mask = segmenter.segment_by_color(image, color)
        objects, _ = segmenter.find_color_objects(image, color, min_area=500)
        result = segmenter.visualize_segmentation(image, mask, objects, color)
        
        axes[i + 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"{color.capitalize()} Segmentation\n({len(objects)} objects)", 
                             fontsize=12, fontweight='bold')
        axes[i + 1].axis('off')
        
        print(f"{color.capitalize()}: {len(objects)} objects detected")
    
    plt.tight_layout()
    plt.show()

def demo_watershed_segmentation():
    """
    Demo Watershed segmentation
    """
    print("\n" + "=" * 50)
    print("WATERSHED SEGMENTATION DEMO")
    print("=" * 50)
    
    # Create test image with touching objects
    image = create_test_image()
    
    # Initialize watershed segmenter
    segmenter = WatershedSegmentation()
    
    # Perform segmentation
    results = segmenter.separate_touching_objects(image)
    
    # Visualize results
    fig = segmenter.visualize_watershed_results(image, results)
    plt.show()
    
    print(f"Found {len(results['objects'])} objects using Watershed algorithm")
    for i, obj in enumerate(results["objects"]):
        print(f"Object {i+1}: Area={obj['area']:.0f} px², Centroid=({obj['centroid'][0]}, {obj['centroid'][1]})")

def demo_advanced_color_analysis():
    """
    Demo advanced color analysis
    """
    print("\n" + "=" * 50)
    print("ADVANCED COLOR ANALYSIS DEMO")
    print("=" * 50)
    
    # Create test image
    image = create_comprehensive_test_image()
    
    # Initialize analyzer
    analyzer = AdvancedColorAnalysis()
    
    # Create comprehensive visualization
    fig = analyzer.create_color_analysis_visualization(image)
    plt.show()
    
    # Print detailed analysis
    print("Color Analysis Results:")
    print("-" * 30)
    
    for space in ['BGR', 'HSV', 'LAB']:
        stats = analyzer.analyze_color_distribution(image, space)
        print(f"\n{space} Color Space Analysis:")
        print(f"  Mean values: {stats['mean']}")
        print(f"  Std deviation: {stats['std']}")
        print(f"  Dominant colors found: {len(stats['dominant_colors'])}")
        
        for i, (color, freq) in enumerate(stats['dominant_colors'][:3]):
            print(f"    Color {i+1}: {color} (frequency: {freq})")
    
    # Color harmony analysis
    harmony = analyzer.analyze_color_harmony(image)
    print(f"\nColor Harmony Analysis:")
    print(f"  Harmony type: {harmony['harmony_type']}")
    print(f"  Confidence: {harmony['confidence']:.2f}")
    print(f"  Dominant hues: {harmony['dominant_hues']}")

def demo_color_tracking():
    """
    Demo color tracking (interactive)
    """
    print("\n" + "=" * 50)
    print("COLOR TRACKING DEMO")
    print("=" * 50)
    print("This demo requires a webcam.")
    print("Press 'q' to quit, 'r' to reset tracking.")
    
    try:
        demo_color_tracking()
    except Exception as e:
        print(f"Color tracking demo failed: {e}")
        print("Make sure you have a webcam connected.")

def create_comparison_visualization():
    """
    Create a comparison visualization of all techniques
    """
    print("\n" + "=" * 50)
    print("TECHNIQUE COMPARISON")
    print("=" * 50)
    
    # Create test image
    image = create_comprehensive_test_image()
    
    # Initialize all analyzers
    hsv_segmenter = HSVSegmentation()
    watershed_segmenter = WatershedSegmentation()
    color_analyzer = AdvancedColorAnalysis()
    
    # Perform different analyses
    # HSV segmentation for red objects
    red_mask = hsv_segmenter.segment_by_color(image, "red")
    red_objects, _ = hsv_segmenter.find_color_objects(image, "red", min_area=500)
    red_result = hsv_segmenter.visualize_segmentation(image, red_mask, red_objects, "red")
    
    # Watershed segmentation
    watershed_results = watershed_segmenter.separate_touching_objects(image)
    
    # Color clustering
    clustered, _ = color_analyzer.segment_by_color_clustering(image, 6)
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # HSV Segmentation
    axes[0, 1].imshow(cv2.cvtColor(red_result, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"HSV Segmentation (Red)\n{len(red_objects)} objects", 
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Watershed Segmentation
    axes[0, 2].imshow(cv2.cvtColor(watershed_results['segmented'], cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"Watershed Segmentation\n{len(watershed_results['objects'])} objects", 
                        fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Color Clustering
    axes[1, 0].imshow(cv2.cvtColor(clustered, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Color Clustering (6 clusters)", 
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Color Palette
    palette = color_analyzer.create_color_palette(image, 8)
    axes[1, 1].imshow(cv2.cvtColor(palette, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Dominant Color Palette", 
                        fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Color Distribution
    data = image.reshape(-1, 3)
    axes[1, 2].hist(data[:, 0], bins=50, alpha=0.7, color='red', label='Blue')
    axes[1, 2].hist(data[:, 1], bins=50, alpha=0.7, color='green', label='Green')
    axes[1, 2].hist(data[:, 2], bins=50, alpha=0.7, color='blue', label='Red')
    axes[1, 2].set_title("Color Distribution", fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].set_xlabel("Pixel Value")
    axes[1, 2].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main demo function
    """
    print("COLOR SPACE ANALYSIS - COMPREHENSIVE DEMO")
    print("=" * 60)
    print("This demo showcases various color analysis techniques:")
    print("1. HSV Color Segmentation")
    print("2. Watershed Algorithm for Object Separation")
    print("3. Advanced Color Analysis and Clustering")
    print("4. Color Tracking (Interactive)")
    print("5. Technique Comparison")
    print("=" * 60)
    
    while True:
        print("\nSelect a demo to run:")
        print("1. HSV Color Segmentation")
        print("2. Watershed Segmentation")
        print("3. Advanced Color Analysis")
        print("4. Color Tracking (Webcam)")
        print("5. Technique Comparison")
        print("6. Run All Demos")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '0':
            print("Exiting demo. Goodbye!")
            break
        elif choice == '1':
            demo_hsv_segmentation()
        elif choice == '2':
            demo_watershed_segmentation()
        elif choice == '3':
            demo_advanced_color_analysis()
        elif choice == '4':
            demo_color_tracking()
        elif choice == '5':
            create_comparison_visualization()
        elif choice == '6':
            print("Running all demos...")
            demo_hsv_segmentation()
            demo_watershed_segmentation()
            demo_advanced_color_analysis()
            create_comparison_visualization()
            print("\nAll demos completed!")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
