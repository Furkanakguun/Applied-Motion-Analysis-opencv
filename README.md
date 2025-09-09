# Applied Motion Analysis with OpenCV

A comprehensive computer vision application for motion analysis using OpenCV. Features two main modules: ball drop motion analysis and basketball shot motion analysis.

## Project Overview

This project is an OpenCV-based system for sports analysis and motion tracking. It uses advanced computer vision techniques including real-world coordinate transformation, marker tracking, and temporal analysis.

## Project Structure

```
Applied-Motion-Analysis-opencv/
├── ball_drop_motion/           # Ball drop motion analysis
│   ├── all_images_2024.py     # Main analysis script
│   ├── best_thresh.py         # Threshold optimization
│   ├── imageMoments.py        # Moment calculation
│   └── *.jpg                  # Analysis images
├── basketball_shot/           # Basketball shot analysis
│   ├── basketball_shot_motion_analysis.py          # Main analysis
│   ├── basketball_shot_motion_analysis_calibration.py  # Calibrated analysis
│   ├── extract_calibration_points.py              # Calibration points
│   ├── images/                # 266 basketball shot frames
│   └── kalibrasyon.jpg        # Calibration image
├── opencv_basics_demo.py      # OpenCV basic functions
├── basketball.jpg             # Demo image
└── LICENSE                    # MIT License
```

## Features

### Ball Drop Motion Analysis
- Image processing with thresholding and median filtering
- Contour detection and analysis
- Moment calculation for centroid detection
- Batch processing of 30 images

### Basketball Shot Analysis
- Calibration system for pixel-to-real-world coordinate conversion
- Marker tracking for 4 body points (shoulder, elbow, wrist, middle finger)
- Temporal tracking across frames
- ROI optimization by cropping 20% from sides
- Real-world coordinate measurements

## Installation

### Requirements
```bash
pip install opencv-python numpy scipy matplotlib
```

### Usage

#### Ball Drop Analysis
```python
cd ball_drop_motion
python all_images_2024.py
```

#### Basketball Shot Analysis
```python
cd basketball_shot
python extract_calibration_points.py
python basketball_shot_motion_analysis_calibration.py
```

#### OpenCV Demo
```python
python opencv_basics_demo.py
```

## Technical Details

### Ball Drop Motion
- Threshold Value: 127
- Filter: Median Blur (3x3)
- Contour Method: RETR_EXTERNAL, CHAIN_APPROX_NONE

### Basketball Shot Analysis
- ROI Margin: 20% from both sides
- Threshold: 230
- Min/Max Area: 5-200 pixels
- Marker Count: 4 body points
- Frame Count: 266
- Coordinate System: Real-world coordinates (cm)

## Calibration System

8-point calibration system for basketball analysis:
- Image Points: Marker positions in calibration image
- World Points: Real-world coordinates (cm)
- Transform: 2D Conformal Transformation (Affine)

## Marker Colors

- Red: Shoulder
- Green: Elbow
- Blue: Wrist
- Yellow: Middle Finger

## Outputs

- Annotated images with markers and coordinates
- Coordinate data for each marker (x, y)
- Trajectory lines connecting markers
- Real-world measurements in centimeters

## Future Developments

- Real-time analysis (webcam)
- Velocity and acceleration calculation
- Video output generation
- Web interface
- Database integration
- Machine learning for shot quality assessment
- 3D analysis module

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Developer

**fakgun** - [GitHub](https://github.com/fakgun)
