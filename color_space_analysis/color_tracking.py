"""
Color-Based Object Tracking
Author: fakgun
Description: Real-time color-based object tracking system
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class ColorTracker:
    def __init__(self, max_track_length=64):
        self.max_track_length = max_track_length
        self.track_points = deque(maxlen=max_track_length)
        self.tracking_color = None
        self.tracking_mask = None
        
    def set_tracking_color(self, hsv_range):
        """
        Set the color range for tracking
        
        Args:
            hsv_range: Dictionary with 'lower' and 'upper' HSV values
        """
        self.tracking_color = hsv_range
        
    def detect_color_objects(self, frame, color_range, min_area=100):
        """
        Detect objects of specified color in frame
        
        Args:
            frame: Input frame
            color_range: HSV color range dictionary
            min_area: Minimum area for object detection
            
        Returns:
            objects: List of detected objects
            mask: Binary mask
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        if 'lower1' in color_range and 'upper1' in color_range:
            # Handle red color (wraps around in HSV)
            mask1 = cv2.inRange(hsv, color_range['lower1'], color_range['upper1'])
            mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    objects.append({
                        'centroid': (cx, cy),
                        'area': area,
                        'bbox': (x, y, w, h),
                        'contour': contour
                    })
        
        return objects, mask
    
    def track_object(self, frame, color_range, min_area=100):
        """
        Track the largest object of specified color
        
        Args:
            frame: Input frame
            color_range: HSV color range
            min_area: Minimum area for tracking
            
        Returns:
            tracked_frame: Frame with tracking visualization
            tracking_info: Dictionary with tracking information
        """
        objects, mask = self.detect_color_objects(frame, color_range, min_area)
        
        # Find the largest object
        if objects:
            largest_obj = max(objects, key=lambda x: x['area'])
            centroid = largest_obj['centroid']
            
            # Add to tracking points
            self.track_points.append(centroid)
            
            # Draw tracking visualization
            tracked_frame = frame.copy()
            
            # Draw bounding box
            x, y, w, h = largest_obj['bbox']
            cv2.rectangle(tracked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw centroid
            cv2.circle(tracked_frame, centroid, 5, (0, 0, 255), -1)
            
            # Draw tracking trail
            if len(self.track_points) > 1:
                points = np.array(self.track_points, dtype=np.int32)
                cv2.polylines(tracked_frame, [points], False, (255, 0, 0), 2)
            
            # Draw mask overlay
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = [0, 255, 0]
            tracked_frame = cv2.addWeighted(tracked_frame, 0.8, colored_mask, 0.2, 0)
            
            tracking_info = {
                'centroid': centroid,
                'area': largest_obj['area'],
                'bbox': largest_obj['bbox'],
                'track_length': len(self.track_points),
                'velocity': self.calculate_velocity()
            }
        else:
            tracked_frame = frame.copy()
            tracking_info = None
        
        return tracked_frame, tracking_info
    
    def calculate_velocity(self):
        """
        Calculate velocity from tracking points
        
        Returns:
            velocity: (vx, vy) velocity in pixels per frame
        """
        if len(self.track_points) < 2:
            return (0, 0)
        
        # Calculate velocity from last two points
        p1 = self.track_points[-2]
        p2 = self.track_points[-1]
        
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        
        return (vx, vy)
    
    def reset_tracking(self):
        """Reset tracking data"""
        self.track_points.clear()

def create_color_picker():
    """
    Interactive color picker for HSV range selection
    """
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get HSV value at clicked point
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_value = hsv[y, x]
            
            print(f"Clicked HSV value: {hsv_value}")
            
            # Create range around the clicked value
            h, s, v = hsv_value
            lower = np.array([max(0, h - 10), max(0, s - 50), max(0, v - 50)])
            upper = np.array([min(179, h + 10), min(255, s + 50), min(255, v + 50)])
            
            print(f"Suggested range:")
            print(f"Lower: {lower}")
            print(f"Upper: {upper}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None, None
    
    print("Click on a color to get HSV range. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show HSV values
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.putText(frame, "Click on color to get HSV range", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Color Picker', frame)
        cv2.setMouseCallback('Color Picker', mouse_callback)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def demo_color_tracking():
    """
    Demo function for color tracking
    """
    # Predefined color ranges
    color_ranges = {
        'red': {
            'lower1': np.array([0, 50, 50]),
            'upper1': np.array([10, 255, 255]),
            'lower2': np.array([170, 50, 50]),
            'upper2': np.array([180, 255, 255])
        },
        'green': {
            'lower': np.array([40, 50, 50]),
            'upper': np.array([80, 255, 255])
        },
        'blue': {
            'lower': np.array([100, 50, 50]),
            'upper': np.array([130, 255, 255])
        }
    }
    
    # Initialize tracker
    tracker = ColorTracker()
    
    # Select color to track
    print("Available colors: red, green, blue")
    color_choice = input("Enter color to track: ").lower()
    
    if color_choice not in color_ranges:
        print("Invalid color choice. Using red.")
        color_choice = 'red'
    
    tracker.set_tracking_color(color_ranges[color_choice])
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print(f"Tracking {color_choice} objects. Press 'q' to quit, 'r' to reset tracking.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track objects
        tracked_frame, tracking_info = tracker.track_object(frame, color_ranges[color_choice])
        
        # Display tracking information
        if tracking_info:
            info_text = f"Area: {tracking_info['area']:.0f}, Velocity: {tracking_info['velocity']}"
            cv2.putText(tracked_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Color Tracking', tracked_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset_tracking()
            print("Tracking reset")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo_color_tracking()
