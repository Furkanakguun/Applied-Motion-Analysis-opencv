# Color Space Analysis Module

This module provides comprehensive color space analysis and segmentation capabilities using OpenCV and advanced computer vision techniques.

## Features

### 1. HSV Color Segmentation (`hsv_segmentation.py`)
- **Color-based object detection** using HSV color space
- **Predefined color ranges** for common colors (red, green, blue, yellow, orange)
- **Object property extraction** (area, centroid, bounding box)
- **Morphological operations** for noise reduction
- **Visualization tools** for segmentation results

### 2. Watershed Algorithm (`watershed_segmentation.py`)
- **Object separation** for touching objects
- **Distance transform** for marker detection
- **Peak detection** using local maxima
- **Morphological preprocessing** for better results
- **Object analysis** with detailed properties

### 3. Color Tracking (`color_tracking.py`)
- **Real-time color-based tracking** using webcam
- **Interactive color picker** for HSV range selection
- **Trajectory visualization** with tracking trails
- **Velocity calculation** from tracking points
- **Multiple color support** with predefined ranges

### 4. Advanced Color Analysis (`advanced_color_analysis.py`)
- **Multiple color space conversion** (BGR, HSV, LAB, YUV, XYZ)
- **Color clustering** using K-means algorithm
- **Dominant color extraction** and palette creation
- **Color harmony analysis** and classification
- **Comprehensive visualization** tools

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic HSV Segmentation
```python
from hsv_segmentation import HSVSegmentation

# Initialize segmenter
segmenter = HSVSegmentation()

# Load image
image = cv2.imread('your_image.jpg')

# Segment red objects
mask = segmenter.segment_by_color(image, "red")
objects, _ = segmenter.find_color_objects(image, "red", min_area=100)

# Visualize results
result = segmenter.visualize_segmentation(image, mask, objects, "red")
cv2.imshow('Result', result)
cv2.waitKey(0)
```

### Watershed Segmentation
```python
from watershed_segmentation import WatershedSegmentation

# Initialize segmenter
segmenter = WatershedSegmentation()

# Load image
image = cv2.imread('touching_objects.jpg')

# Separate touching objects
results = segmenter.separate_touching_objects(image)

# Visualize results
fig = segmenter.visualize_watershed_results(image, results)
plt.show()
```

### Color Tracking
```python
from color_tracking import ColorTracker

# Initialize tracker
tracker = ColorTracker()

# Set color range (example for red)
red_range = {
    'lower1': np.array([0, 50, 50]),
    'upper1': np.array([10, 255, 255]),
    'lower2': np.array([170, 50, 50]),
    'upper2': np.array([180, 255, 255])
}

# Track objects in real-time
tracker.set_tracking_color(red_range)
# Use with webcam feed...
```

### Advanced Color Analysis
```python
from advanced_color_analysis import AdvancedColorAnalysis

# Initialize analyzer
analyzer = AdvancedColorAnalysis()

# Load image
image = cv2.imread('colorful_image.jpg')

# Create comprehensive analysis
fig = analyzer.create_color_analysis_visualization(image)
plt.show()

# Get color statistics
stats = analyzer.analyze_color_distribution(image, 'HSV')
print(f"Mean HSV values: {stats['mean']}")
```

## Running the Demo

### Interactive Demo
```bash
python main_demo.py
```

This will show a menu with different demo options:
1. HSV Color Segmentation
2. Watershed Segmentation  
3. Advanced Color Analysis
4. Color Tracking (Webcam)
5. Technique Comparison
6. Run All Demos

### Individual Demos
```bash
# HSV Segmentation
python hsv_segmentation.py

# Watershed Segmentation
python watershed_segmentation.py

# Color Tracking
python color_tracking.py

# Advanced Analysis
python advanced_color_analysis.py
```

## Color Ranges

### Predefined HSV Ranges
- **Red**: H=[0-10, 170-180], S=[50-255], V=[50-255]
- **Green**: H=[40-80], S=[50-255], V=[50-255]
- **Blue**: H=[100-130], S=[50-255], V=[50-255]
- **Yellow**: H=[20-30], S=[50-255], V=[50-255]
- **Orange**: H=[10-20], S=[50-255], V=[50-255]

### Custom Color Range
```python
# Define custom HSV range
custom_range = {
    'lower': np.array([h_min, s_min, v_min]),
    'upper': np.array([h_max, s_max, v_max])
}

# Use with segmenter
mask = segmenter.create_hsv_mask(image, custom_range['lower'], custom_range['upper'])
```

## Applications

### Sports Analysis
- **Ball tracking** in various sports
- **Player movement** analysis
- **Equipment detection** and tracking

### Medical Imaging
- **Tissue segmentation** in medical images
- **Color-based diagnosis** support
- **Pathology detection**

### Industrial Applications
- **Quality control** in manufacturing
- **Object sorting** by color
- **Defect detection** in products

### Computer Vision Research
- **Color space comparison** studies
- **Segmentation algorithm** evaluation
- **Feature extraction** for machine learning

## Technical Details

### HSV Color Space
- **Hue (H)**: Color type (0-179 in OpenCV)
- **Saturation (S)**: Color intensity (0-255)
- **Value (V)**: Brightness (0-255)

### Watershed Algorithm
- **Distance Transform**: Calculates distance to nearest boundary
- **Peak Detection**: Finds local maxima for markers
- **Morphological Operations**: Preprocessing for better results

### Color Clustering
- **K-means Algorithm**: Groups similar colors
- **Feature Extraction**: Dominant color identification
- **Harmony Analysis**: Color relationship classification

## Performance Tips

1. **ROI Selection**: Crop images to relevant regions
2. **Resolution**: Use appropriate image resolution
3. **Preprocessing**: Apply noise reduction filters
4. **Threshold Tuning**: Adjust parameters for your use case
5. **Batch Processing**: Process multiple images efficiently

## Troubleshooting

### Common Issues
- **No objects detected**: Adjust color ranges or minimum area
- **Too many false positives**: Increase minimum area threshold
- **Poor segmentation**: Check lighting conditions and color contrast
- **Slow performance**: Reduce image resolution or use ROI

### Debug Tips
- Use `cv2.imshow()` to visualize intermediate results
- Print HSV values of target colors
- Adjust morphological kernel sizes
- Test with different color spaces

## Contributing

Feel free to contribute to this module by:
- Adding new color spaces
- Improving segmentation algorithms
- Adding new visualization tools
- Optimizing performance
- Adding new applications

## License

This module is part of the Applied Motion Analysis project and follows the same MIT license.
