"""
Advanced Color Space Analysis
Author: fakgun
Description: Advanced color analysis including multiple color spaces and clustering
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class AdvancedColorAnalysis:
    def __init__(self):
        self.color_spaces = ['BGR', 'HSV', 'LAB', 'YUV', 'XYZ']
        
    def convert_to_multiple_spaces(self, image):
        """
        Convert image to multiple color spaces
        
        Args:
            image: Input BGR image
            
        Returns:
            spaces: Dictionary with different color space representations
        """
        spaces = {}
        
        # BGR (original)
        spaces['BGR'] = image.copy()
        
        # HSV
        spaces['HSV'] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # LAB
        spaces['LAB'] = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # YUV
        spaces['YUV'] = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # XYZ
        spaces['XYZ'] = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
        
        return spaces
    
    def analyze_color_distribution(self, image, color_space='BGR'):
        """
        Analyze color distribution in specified color space
        
        Args:
            image: Input image
            color_space: Color space to analyze
            
        Returns:
            stats: Dictionary with color statistics
        """
        if color_space == 'BGR':
            data = image.reshape(-1, 3)
        else:
            spaces = self.convert_to_multiple_spaces(image)
            data = spaces[color_space].reshape(-1, 3)
        
        stats = {}
        
        # Basic statistics
        stats['mean'] = np.mean(data, axis=0)
        stats['std'] = np.std(data, axis=0)
        stats['min'] = np.min(data, axis=0)
        stats['max'] = np.max(data, axis=0)
        
        # Dominant colors
        stats['dominant_colors'] = self.find_dominant_colors(data, n_colors=5)
        
        return stats
    
    def find_dominant_colors(self, data, n_colors=5):
        """
        Find dominant colors using K-means clustering
        
        Args:
            data: Color data (N x 3)
            n_colors: Number of dominant colors to find
            
        Returns:
            colors: List of dominant colors
        """
        # Resample data for faster processing
        if len(data) > 10000:
            indices = np.random.choice(len(data), 10000, replace=False)
            data_sample = data[indices]
        else:
            data_sample = data
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(data_sample)
        
        # Get cluster centers
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get color frequencies
        labels = kmeans.labels_
        frequencies = [np.sum(labels == i) for i in range(n_colors)]
        
        # Sort by frequency
        sorted_indices = np.argsort(frequencies)[::-1]
        colors = colors[sorted_indices]
        frequencies = [frequencies[i] for i in sorted_indices]
        
        return list(zip(colors, frequencies))
    
    def create_color_palette(self, image, n_colors=8):
        """
        Create a color palette from image
        
        Args:
            image: Input image
            n_colors: Number of colors in palette
            
        Returns:
            palette: Color palette image
        """
        data = image.reshape(-1, 3)
        dominant_colors = self.find_dominant_colors(data, n_colors)
        
        # Create palette image
        palette_height = 100
        palette_width = n_colors * 100
        palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
        
        for i, (color, freq) in enumerate(dominant_colors):
            start_x = i * 100
            end_x = (i + 1) * 100
            palette[:, start_x:end_x] = color
        
        return palette
    
    def segment_by_color_clustering(self, image, n_clusters=5):
        """
        Segment image using color clustering
        
        Args:
            image: Input image
            n_clusters: Number of clusters
            
        Returns:
            segmented: Segmented image
            labels: Cluster labels
        """
        data = image.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # Reshape labels back to image shape
        labels = labels.reshape(image.shape[:2])
        
        # Create segmented image
        segmented = np.zeros_like(image)
        for i in range(n_clusters):
            mask = labels == i
            segmented[mask] = kmeans.cluster_centers_[i]
        
        return segmented, labels
    
    def analyze_color_harmony(self, image):
        """
        Analyze color harmony in image
        
        Args:
            image: Input image
            
        Returns:
            harmony_info: Dictionary with harmony analysis
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_values = hsv[:, :, 0].flatten()
        
        # Remove black/white pixels (saturation < 30 or value < 30)
        s_values = hsv[:, :, 1].flatten()
        v_values = hsv[:, :, 2].flatten()
        
        valid_pixels = (s_values > 30) & (v_values > 30)
        h_valid = h_values[valid_pixels]
        
        if len(h_valid) == 0:
            return {"harmony_type": "No valid colors", "confidence": 0}
        
        # Analyze hue distribution
        hue_hist, _ = np.histogram(h_valid, bins=36, range=(0, 180))
        
        # Find dominant hue ranges
        dominant_hues = np.argsort(hue_hist)[-3:][::-1]
        
        # Determine harmony type
        harmony_info = self.classify_color_harmony(dominant_hues, hue_hist)
        
        return harmony_info
    
    def classify_color_harmony(self, dominant_hues, hue_hist):
        """
        Classify color harmony type
        
        Args:
            dominant_hues: Indices of dominant hues
            hue_hist: Histogram of hue values
            
        Returns:
            harmony_info: Dictionary with harmony classification
        """
        # Convert hue indices to actual hue values
        hue_values = dominant_hues * 5  # Each bin represents 5 degrees
        
        # Calculate hue differences
        if len(hue_values) >= 2:
            diff1 = abs(hue_values[0] - hue_values[1])
            diff2 = abs(hue_values[1] - hue_values[2]) if len(hue_values) >= 3 else 0
            diff3 = abs(hue_values[0] - hue_values[2]) if len(hue_values) >= 3 else 0
            
            # Classify harmony
            if diff1 < 30:
                harmony_type = "Monochromatic"
                confidence = 0.9
            elif 60 <= diff1 <= 120:
                harmony_type = "Complementary"
                confidence = 0.8
            elif 120 <= diff1 <= 180:
                harmony_type = "Triadic"
                confidence = 0.7
            else:
                harmony_type = "Analogous"
                confidence = 0.6
        else:
            harmony_type = "Single Color"
            confidence = 0.5
        
        return {
            "harmony_type": harmony_type,
            "confidence": confidence,
            "dominant_hues": hue_values.tolist()
        }
    
    def create_color_analysis_visualization(self, image):
        """
        Create comprehensive color analysis visualization
        
        Args:
            image: Input image
            
        Returns:
            fig: Matplotlib figure with analysis
        """
        # Analyze in multiple color spaces
        spaces = self.convert_to_multiple_spaces(image)
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Color spaces
        space_names = ['HSV', 'LAB', 'YUV']
        for i, space in enumerate(space_names):
            if space == 'HSV':
                # For HSV, show H channel in color
                hsv = spaces[space]
                h_channel = hsv[:, :, 0]
                axes[i + 1].imshow(h_channel, cmap='hsv')
            else:
                axes[i + 1].imshow(spaces[space])
            axes[i + 1].set_title(f"{space} Color Space")
            axes[i + 1].axis('off')
        
        # Color palette
        palette = self.create_color_palette(image, 8)
        axes[4].imshow(cv2.cvtColor(palette, cv2.COLOR_BGR2RGB))
        axes[4].set_title("Color Palette")
        axes[4].axis('off')
        
        # Segmented image
        segmented, _ = self.segment_by_color_clustering(image, 5)
        axes[5].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        axes[5].set_title("Color Clustering (5 clusters)")
        axes[5].axis('off')
        
        # Color distribution
        data = image.reshape(-1, 3)
        axes[6].hist(data[:, 0], bins=50, alpha=0.7, color='red', label='Blue')
        axes[6].hist(data[:, 1], bins=50, alpha=0.7, color='green', label='Green')
        axes[6].hist(data[:, 2], bins=50, alpha=0.7, color='blue', label='Red')
        axes[6].set_title("Color Distribution")
        axes[6].legend()
        
        # HSV histogram
        hsv = spaces['HSV']
        axes[7].hist(hsv[:, :, 0].flatten(), bins=36, alpha=0.7, color='purple')
        axes[7].set_title("Hue Distribution")
        axes[7].set_xlabel("Hue (0-180)")
        
        # Color harmony analysis
        harmony_info = self.analyze_color_harmony(image)
        axes[8].text(0.1, 0.5, f"Harmony Type: {harmony_info['harmony_type']}\n"
                               f"Confidence: {harmony_info['confidence']:.2f}\n"
                               f"Dominant Hues: {harmony_info['dominant_hues']}",
                    transform=axes[8].transAxes, fontsize=12,
                    verticalalignment='center')
        axes[8].set_title("Color Harmony Analysis")
        axes[8].axis('off')
        
        plt.tight_layout()
        return fig

def demo_advanced_color_analysis():
    """
    Demo function for advanced color analysis
    """
    # Create a colorful test image
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add colorful regions
    cv2.rectangle(image, (0, 0), (100, 100), (255, 0, 0), -1)      # Blue
    cv2.rectangle(image, (100, 0), (200, 100), (0, 255, 0), -1)    # Green
    cv2.rectangle(image, (200, 0), (300, 100), (0, 0, 255), -1)    # Red
    cv2.rectangle(image, (300, 0), (400, 100), (255, 255, 0), -1)  # Cyan
    
    cv2.rectangle(image, (0, 100), (100, 200), (255, 0, 255), -1)  # Magenta
    cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 255), -1) # Yellow
    cv2.rectangle(image, (200, 100), (300, 200), (128, 128, 128), -1) # Gray
    cv2.rectangle(image, (300, 100), (400, 200), (64, 128, 192), -1)  # Custom
    
    # Add some gradients
    for i in range(100):
        color = (i * 2, 128, 255 - i * 2)
        cv2.line(image, (0, 200 + i), (400, 200 + i), color, 1)
    
    # Initialize analyzer
    analyzer = AdvancedColorAnalysis()
    
    # Create visualization
    fig = analyzer.create_color_analysis_visualization(image)
    plt.show()
    
    # Print analysis results
    print("Color Analysis Results:")
    print("=" * 30)
    
    for space in ['BGR', 'HSV', 'LAB']:
        stats = analyzer.analyze_color_distribution(image, space)
        print(f"\n{space} Color Space:")
        print(f"Mean: {stats['mean']}")
        print(f"Std: {stats['std']}")
        print(f"Dominant colors: {len(stats['dominant_colors'])}")

if __name__ == "__main__":
    demo_advanced_color_analysis()
