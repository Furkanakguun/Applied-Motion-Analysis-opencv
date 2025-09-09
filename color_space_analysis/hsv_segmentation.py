"""
HSV Color Space Segmentation
Author: fakgun
Description: HSV color space analysis and segmentation for object detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class HSVSegmentation:
    def __init__(self):
        self.lower_hsv = np.array([0, 0, 0])
        self.upper_hsv = np.array([179, 255, 255])
        
    def create_hsv_mask(self, image, lower_hsv, upper_hsv):
        """
        Create HSV mask for color segmentation
        
        Args:
            image: Input BGR image
            lower_hsv: Lower HSV threshold values
            upper_hsv: Upper HSV threshold values
            
        Returns:
            mask: Binary mask
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        return mask
    
    def segment_by_color(self, image, color_name="red"):
        """
        Segment image by predefined color ranges
        
        Args:
            image: Input BGR image
            color_name: Color to segment ("red", "green", "blue", "yellow", "orange")
            
        Returns:
            mask: Binary mask for the specified color
        """
        color_ranges = {
            "red": {
                "lower1": np.array([0, 50, 50]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([170, 50, 50]),
                "upper2": np.array([180, 255, 255])
            },
            "green": {
                "lower": np.array([40, 50, 50]),
                "upper": np.array([80, 255, 255])
            },
            "blue": {
                "lower": np.array([100, 50, 50]),
                "upper": np.array([130, 255, 255])
            },
            "yellow": {
                "lower": np.array([20, 50, 50]),
                "upper": np.array([30, 255, 255])
            },
            "orange": {
                "lower": np.array([10, 50, 50]),
                "upper": np.array([20, 255, 255])
            }
        }
        
        if color_name not in color_ranges:
            raise ValueError(f"Color '{color_name}' not supported. Available: {list(color_ranges.keys())}")
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if color_name == "red":
            # Red color wraps around in HSV, so we need two ranges
            mask1 = cv2.inRange(hsv, color_ranges[color_name]["lower1"], color_ranges[color_name]["upper1"])
            mask2 = cv2.inRange(hsv, color_ranges[color_name]["lower2"], color_ranges[color_name]["upper2"])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, color_ranges[color_name]["lower"], color_ranges[color_name]["upper"])
        
        return mask
    
    def find_color_objects(self, image, color_name="red", min_area=100):
        """
        Find objects of specific color and return their properties
        
        Args:
            image: Input BGR image
            color_name: Color to detect
            min_area: Minimum area for object detection
            
        Returns:
            objects: List of dictionaries with object properties
        """
        mask = self.segment_by_color(image, color_name)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_area:
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                objects.append({
                    "id": i,
                    "area": area,
                    "bbox": (x, y, w, h),
                    "centroid": (cx, cy),
                    "contour": contour
                })
        
        return objects, mask
    
    def visualize_segmentation(self, image, mask, objects=None, color_name="red"):
        """
        Visualize segmentation results
        
        Args:
            image: Original image
            mask: Binary mask
            objects: List of detected objects
            color_name: Name of segmented color
            
        Returns:
            result_image: Annotated result image
        """
        # Create result image
        result = image.copy()
        
        # Apply mask as colored overlay
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 255, 0]  # Green overlay for detected areas
        result = cv2.addWeighted(result, 0.7, colored_mask, 0.3, 0)
        
        # Draw bounding boxes and labels for detected objects
        if objects:
            for obj in objects:
                x, y, w, h = obj["bbox"]
                cx, cy = obj["centroid"]
                
                # Draw bounding rectangle
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Draw centroid
                cv2.circle(result, (cx, cy), 5, (255, 0, 0), -1)
                
                # Draw label
                label = f"{color_name} {obj['id']}: {obj['area']:.0f}"
                cv2.putText(result, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result

def demo_hsv_segmentation():
    """
    Demo function for HSV segmentation
    """
    # Create a sample image with different colored objects
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Draw colored circles
    cv2.circle(image, (150, 150), 50, (0, 0, 255), -1)  # Red
    cv2.circle(image, (300, 150), 50, (0, 255, 0), -1)  # Green
    cv2.circle(image, (450, 150), 50, (255, 0, 0), -1)  # Blue
    cv2.circle(image, (225, 300), 50, (0, 255, 255), -1)  # Yellow
    cv2.circle(image, (375, 300), 50, (0, 165, 255), -1)  # Orange
    
    # Initialize segmenter
    segmenter = HSVSegmentation()
    
    # Test different colors
    colors = ["red", "green", "blue", "yellow", "orange"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Segment each color
    for i, color in enumerate(colors):
        mask = segmenter.segment_by_color(image, color)
        objects, _ = segmenter.find_color_objects(image, color, min_area=100)
        result = segmenter.visualize_segmentation(image, mask, objects, color)
        
        axes[i + 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"{color.capitalize()} Segmentation")
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_hsv_segmentation()
