"""
Watershed Algorithm for Object Separation
Author: fakgun
Description: Watershed algorithm implementation for separating touching objects
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_maxima
from skimage.segmentation import watershed

class WatershedSegmentation:
    def __init__(self):
        self.min_distance = 20
        self.threshold_abs = 0.3
        
    def preprocess_for_watershed(self, image):
        """
        Preprocess image for watershed algorithm
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            processed: Preprocessed grayscale image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        return processed
    
    def find_markers(self, image, min_distance=20, threshold_abs=0.3):
        """
        Find markers for watershed algorithm using distance transform
        
        Args:
            image: Preprocessed grayscale image
            min_distance: Minimum distance between markers
            threshold_abs: Threshold for peak detection
            
        Returns:
            markers: Marker image
            num_markers: Number of found markers
        """
        # Apply threshold to create binary image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Compute distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Find local maxima
        local_maxima = peak_local_maxima(
            dist_transform, 
            min_distance=min_distance, 
            threshold_abs=threshold_abs * dist_transform.max()
        )
        
        # Create marker image
        markers = np.zeros_like(image, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Apply watershed to markers
        markers = watershed(-dist_transform, markers, mask=binary)
        
        return markers, len(local_maxima)
    
    def watershed_segment(self, image, markers):
        """
        Apply watershed algorithm for object separation
        
        Args:
            image: Original image
            markers: Marker image from find_markers
            
        Returns:
            labels: Watershed labels
            segmented: Segmented image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply watershed
        labels = watershed(gray, markers)
        
        # Create segmented image
        segmented = np.zeros_like(image)
        for label in np.unique(labels):
            if label == 0:  # Background
                continue
            mask = labels == label
            segmented[mask] = np.random.randint(0, 255, 3)
        
        return labels, segmented
    
    def separate_touching_objects(self, image, min_distance=20, threshold_abs=0.3):
        """
        Complete pipeline for separating touching objects
        
        Args:
            image: Input image
            min_distance: Minimum distance between markers
            threshold_abs: Threshold for peak detection
            
        Returns:
            results: Dictionary with segmentation results
        """
        # Preprocess image
        processed = self.preprocess_for_watershed(image)
        
        # Find markers
        markers, num_markers = self.find_markers(processed, min_distance, threshold_abs)
        
        # Apply watershed
        labels, segmented = self.watershed_segment(image, markers)
        
        # Analyze segments
        objects = self.analyze_segments(labels, image)
        
        return {
            "processed": processed,
            "markers": markers,
            "labels": labels,
            "segmented": segmented,
            "objects": objects,
            "num_objects": num_markers
        }
    
    def analyze_segments(self, labels, original_image):
        """
        Analyze segmented objects and extract properties
        
        Args:
            labels: Watershed labels
            original_image: Original image for color analysis
            
        Returns:
            objects: List of object properties
        """
        objects = []
        
        for label in np.unique(labels):
            if label == 0:  # Background
                continue
                
            # Create mask for this object
            mask = labels == label
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = contours[0]
                area = cv2.contourArea(contour)
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                # Average color
                if len(original_image.shape) == 3:
                    avg_color = np.mean(original_image[mask], axis=0)
                else:
                    avg_color = np.mean(original_image[mask])
                
                objects.append({
                    "label": label,
                    "area": area,
                    "bbox": (x, y, w, h),
                    "centroid": (cx, cy),
                    "contour": contour,
                    "avg_color": avg_color,
                    "mask": mask
                })
        
        return objects
    
    def visualize_watershed_results(self, original_image, results):
        """
        Visualize watershed segmentation results
        
        Args:
            original_image: Original input image
            results: Results from separate_touching_objects
            
        Returns:
            visualization: Combined visualization image
        """
        # Create subplot visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        if len(original_image.shape) == 3:
            axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        else:
            axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Processed image
        axes[0, 1].imshow(results["processed"], cmap='gray')
        axes[0, 1].set_title("Preprocessed")
        axes[0, 1].axis('off')
        
        # Markers
        axes[0, 2].imshow(results["markers"], cmap='nipy_spectral')
        axes[0, 2].set_title(f"Markers ({results['num_objects']} objects)")
        axes[0, 2].axis('off')
        
        # Segmented image
        if len(original_image.shape) == 3:
            axes[1, 0].imshow(cv2.cvtColor(results["segmented"], cv2.COLOR_BGR2RGB))
        else:
            axes[1, 0].imshow(results["segmented"])
        axes[1, 0].set_title("Watershed Segmentation")
        axes[1, 0].axis('off')
        
        # Overlay on original
        overlay = original_image.copy()
        for obj in results["objects"]:
            x, y, w, h = obj["bbox"]
            cx, cy = obj["centroid"]
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw centroid
            cv2.circle(overlay, (cx, cy), 5, (255, 0, 0), -1)
            
            # Draw label
            cv2.putText(overlay, f"Obj {obj['label']}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if len(overlay.shape) == 3:
            axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        else:
            axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("Detected Objects")
        axes[1, 1].axis('off')
        
        # Object statistics
        axes[1, 2].axis('off')
        stats_text = f"Objects Found: {len(results['objects'])}\n\n"
        for i, obj in enumerate(results["objects"]):
            stats_text += f"Object {obj['label']}:\n"
            stats_text += f"  Area: {obj['area']:.0f} px²\n"
            stats_text += f"  Centroid: ({obj['centroid'][0]}, {obj['centroid'][1]})\n"
            if i < len(results["objects"]) - 1:
                stats_text += "\n"
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title("Object Statistics")
        
        plt.tight_layout()
        return fig

def create_test_image():
    """
    Create a test image with touching objects
    """
    # Create base image
    image = np.ones((300, 400, 3), dtype=np.uint8) * 50
    
    # Draw touching circles
    cv2.circle(image, (100, 100), 40, (255, 100, 100), -1)  # Red-ish
    cv2.circle(image, (140, 100), 40, (100, 255, 100), -1)  # Green-ish
    cv2.circle(image, (180, 100), 40, (100, 100, 255), -1)  # Blue-ish
    
    # Draw touching rectangles
    cv2.rectangle(image, (50, 200), (120, 250), (255, 255, 100), -1)  # Yellow-ish
    cv2.rectangle(image, (100, 200), (170, 250), (255, 100, 255), -1)  # Magenta-ish
    
    # Add some noise
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image

def demo_watershed():
    """
    Demo function for watershed segmentation
    """
    # Create test image
    test_image = create_test_image()
    
    # Initialize watershed segmenter
    segmenter = WatershedSegmentation()
    
    # Perform segmentation
    results = segmenter.separate_touching_objects(test_image)
    
    # Visualize results
    fig = segmenter.visualize_watershed_results(test_image, results)
    plt.show()
    
    print(f"Found {len(results['objects'])} objects")
    for obj in results["objects"]:
        print(f"Object {obj['label']}: Area={obj['area']:.0f}, Centroid={obj['centroid']}")

if __name__ == "__main__":
    demo_watershed()
