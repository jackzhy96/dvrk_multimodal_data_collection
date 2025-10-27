"""
dVRK Data Annotation GUI

A comprehensive PyQt-based GUI for annotating surgical events, phases, and contact detection
in dVRK multi-modal data. This tool provides an intuitive interface for:

- Multi-camera video display with synchronized timeline navigation
- Event, phase, and contact annotation capabilities
- Multi-frame batch labeling functionality
- Auto-play with configurable speed
- Flexible annotation saving with proper folder structure
- PSM contact detection with boolean labels
- Configurable image sizing and display options
"""

import json
import sys
import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import traceback

import cv2
import numpy as np
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QTextEdit, QLineEdit, QListWidget,
    QListWidgetItem, QSplitter, QGroupBox, QGridLayout, QMessageBox,
    QFileDialog, QStatusBar, QProgressBar, QComboBox, QSpinBox,
    QCheckBox, QTabWidget, QScrollArea, QFrame, QButtonGroup,
    QRadioButton, QFormLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor

# Import existing utility functions using the same pattern
proj_root = Path(__file__).resolve().parents[3]
sys.path.append(str(proj_root / "src"))

from dvrk_data_processing.utils.utility import (
    create_folder, glob_sorted_frame, convert_pathlib_type
)



class ConfigLoader:
    """
    Loads and manages configuration using Hydra config system.
    
    Handles loading the config_annotation.yaml file with proper path resolution
    and provides access to all configuration parameters including PSM settings,
    image sizing, and folder structure.
    """
    
    def __init__(self):
        self.config: Optional[DictConfig] = None
        self.config_path: Optional[Path] = None
    
    def load_config(self, config_path: Optional[Path] = None) -> DictConfig:
        """
        Load configuration using Hydra config system.
        
        Args:
            config_path: Optional path to config file. If None, searches for config_annotation.yaml
            
        Returns:
            Loaded configuration as DictConfig
        """
        try:
            if config_path is None:
                # Search for config_annotation.yaml
                config_path = self._find_config_file()
            
            if not config_path or not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            self.config_path = config_path
            
            # Initialize Hydra and load config
            with hydra.initialize_config_dir(config_dir=str(config_path.parent)):
                cfg = hydra.compose(config_name=config_path.stem)
                self.config = cfg
                
                # Resolve any remaining variables in the config
                OmegaConf.resolve(self.config)
                
                return self.config
                
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise
    
    def _find_config_file(self) -> Optional[Path]:
        """
        Search for config_annotation.yaml file in the project.
        
        Returns:
            Path to config file or None if not found
        """
        # Start from current script location and search upwards
        search_paths = [
            Path(__file__).resolve().parent.parent.parent.parent / "config",  # Project root config
            Path(__file__).resolve().parent / "config",  # Local config
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                config_file = search_path / "config_annotation.yaml"
                if config_file.exists():
                    return config_file
        
        return None
    
    def get_image_paths(self) -> Dict[str, Path]:
        """
        Get image paths for all cameras based on configuration.
        
        Returns:
            Dictionary mapping camera names to their image directories
        """
        if not self.config:
            raise ValueError("Configuration not loaded")
        
        image_paths = {}
        
        # Left camera (main stereo camera)
        if hasattr(self.config, 'left_image_path') and self.config.left_image_path:
            image_paths['left'] = Path(self.config.left_image_path)
        
        # Right camera (main stereo camera)  
        if hasattr(self.config, 'right_image_path') and self.config.right_image_path:
            image_paths['right'] = Path(self.config.right_image_path)
        
        # Side cameras
        for i in range(1, 3):  # side_camera_1_path, side_camera_2_path
            attr_name = f'side_camera_{i}_path'
            if hasattr(self.config, attr_name):
                side_path = getattr(self.config, attr_name)
                # Handle both string "None" and actual None values
                if side_path and side_path != "None" and side_path is not None:
                    image_paths[f'side_{i}'] = Path(side_path)
        
        return image_paths
    
    def get_save_folder(self) -> Path:
        """Get the base save folder from configuration."""
        if not self.config:
            raise ValueError("Configuration not loaded")
        
        return Path(self.config.save_folder) if hasattr(self.config, 'save_folder') else Path("./annotation_output")
    
    def get_image_size_config(self) -> Dict[str, Any]:
        """Get image sizing configuration."""
        if not self.config:
            raise ValueError("No configuration loaded - cannot get image size config")
        
        # Convert OmegaConf ListConfig to regular Python lists to avoid array comparison issues
        main_size = getattr(self.config, 'main_image_max_size')
        side_size = getattr(self.config, 'side_image_max_size')
        
        # Convert OmegaConf objects to native Python types
        def convert_to_list(value, default):
            if value is None:
                if default is None:
                    raise ValueError("Image size config is missing and no default provided")
                return default
            elif hasattr(value, '_content'):  # OmegaConf ListConfig
                return OmegaConf.to_container(value)
            elif isinstance(value, (list, tuple)):
                return list(value)
            else:
                if default is None:
                    raise ValueError(f"Invalid image size config type: {type(value)}")
                return default
        
        return {
            'resize_max_size': getattr(self.config, 'resize_max_size', False),
            'main_image_max_size': convert_to_list(main_size, None),
            'side_image_max_size': convert_to_list(side_size, None)
        }
    
    def get_psm_config(self) -> Dict[str, bool]:
        """Get PSM configuration from config file."""
        if not self.config:
            return {'PSM1': True, 'PSM2': True, 'PSM3': False}  # Default
        
        return {
            'PSM1': getattr(self.config, 'enable_PSM1', True),
            'PSM2': getattr(self.config, 'enable_PSM2', True),
            'PSM3': getattr(self.config, 'enable_PSM3', False)
        }
    
    def get_gui_config(self) -> Dict[str, Any]:
        """Get GUI configuration parameters for better scalability."""
        if not self.config or not hasattr(self.config, 'gui_config'):
            # Return default GUI configuration optimized for 1080P (1920x1080)
            # CRITICAL FIX: Ensure FULL window is visible (no parts cut off)
            # Window size must account for: taskbar (~40px) + decorations (~40px) = ~80px total
            # Available space: 1080 - 80 = 1000px maximum safe height
            return {
                'window_width': 1910,  # Slightly wider (user: width has space left)
                'window_height': 1000,  # Safe height to ensure full visibility
                'default_playback_speed_ms': 33,
                'min_playback_speed_ms': 10,
                'max_playback_speed_ms': 1000,
                'image_loader_refresh_ms': 50,
                'auto_save_enabled': True,
                'auto_save_interval_seconds': 300,
                'max_backup_files': 10,
                'max_side_cameras': 2,
                'annotation_list_height': 150
            }
        
        gui_config = self.config.gui_config
        return {
            'window_width': getattr(gui_config, 'window_width', 1910),
            'window_height': getattr(gui_config, 'window_height', 1000),
            'default_playback_speed_ms': getattr(gui_config, 'default_playback_speed_ms', 33),
            'min_playback_speed_ms': getattr(gui_config, 'min_playback_speed_ms', 10),
            'max_playback_speed_ms': getattr(gui_config, 'max_playback_speed_ms', 1000),
            'image_loader_refresh_ms': getattr(gui_config, 'image_loader_refresh_ms', 50),
            'auto_save_enabled': getattr(gui_config, 'auto_save_enabled', True),
            'auto_save_interval_seconds': getattr(gui_config, 'auto_save_interval_seconds', 300),
            'max_backup_files': getattr(gui_config, 'max_backup_files', 10),
            'max_side_cameras': getattr(gui_config, 'max_side_cameras', 2),
            'annotation_list_height': getattr(gui_config, 'annotation_list_height', 150)
        }
    
    def get_annotation_categories(self) -> List[str]:
        """Get configurable annotation categories."""
        if not self.config or not hasattr(self.config, 'annotation_categories'):
            return ["event", "phase", "contact"]  # Default categories
        
        # Convert OmegaConf ListConfig to regular Python list
        categories = self.config.annotation_categories
        if hasattr(categories, '_content'):  # OmegaConf ListConfig
            return OmegaConf.to_container(categories)
        elif isinstance(categories, (list, tuple)):
            return list(categories)
        else:
            return ["event", "phase", "contact"]  # Fallback to defaults
    
    def get_quick_action_labels(self) -> List[str]:
        """Get configurable quick action labels."""
        if not self.config or not hasattr(self.config, 'quick_action_labels'):
            return ["Phase Start", "Phase End", "Event Marker", "Contact Detected", "Contact Lost"]  # Default
        
        # Convert OmegaConf ListConfig to regular Python list
        labels = self.config.quick_action_labels
        if hasattr(labels, '_content'):  # OmegaConf ListConfig
            return OmegaConf.to_container(labels)
        elif isinstance(labels, (list, tuple)):
            return list(labels)
        else:
            return ["Phase Start", "Phase End", "Event Marker", "Contact Detected", "Contact Lost"]  # Fallback


class ImageProcessor(QThread):
    """
    Background thread for loading and processing video frames.
    
    Handles loading images from multiple cameras, applying proper resizing
    based on configuration, and combining them for display according to
    the size constraints from the yaml file.
    """
    
    # Signals for communication with main thread
    images_loaded = pyqtSignal(np.ndarray, str)  # combined_image, frame_info
    loading_error = pyqtSignal(str)  # error_message
    
    def __init__(self, image_paths: Dict[str, Path], frame_files: List[Path], size_config: Dict[str, Any], gui_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.image_paths = image_paths
        self.frame_files = frame_files
        self.size_config = size_config
        self.gui_config = gui_config if gui_config else {'image_loader_refresh_ms': 50}
        self.current_frame_index = 0
        self.target_frame_index = 0
        self.running = True
        
        # Extract size constraints from config and ensure they are regular Python lists
        self.resize_max_size = size_config.get('resize_max_size', False)
        main_size = size_config.get('main_image_max_size')
        side_size = size_config.get('side_image_max_size')
        
        # Convert OmegaConf objects to native Python types to avoid array comparison issues
        def convert_to_list(value, default):
            try:
                if value is None:
                    if default is None:
                        raise ValueError("Image size config is missing and no default provided")
                    return default
                elif hasattr(value, '_content'):  # OmegaConf ListConfig
                    return OmegaConf.to_container(value)
                elif isinstance(value, (list, tuple)):
                    return list(value)
                else:
                    if default is None:
                        raise ValueError(f"Invalid image size config type: {type(value)}")
                    return default
            except Exception as e:
                print(f"Error converting image size config {value}: {e}")
                if default is None:
                    raise ValueError("Invalid image size configuration - no fallback available") from e
                return default
        
        self.main_image_max_size = convert_to_list(main_size, None)
        self.side_image_max_size = convert_to_list(side_size, None)
    
    def set_frame_index(self, index: int):
        """Request loading of specific frame index."""
        self.target_frame_index = max(0, min(index, len(self.frame_files) - 1))
    
    def run(self):
        """Main thread execution - loads images on demand."""
        while self.running:
            if self.target_frame_index != self.current_frame_index:
                try:
                    self.load_frame(self.target_frame_index)
                    self.current_frame_index = self.target_frame_index
                except Exception as e:
                    self.loading_error.emit(f"Error loading frame {self.target_frame_index}: {str(e)}")
            
            # Use configurable refresh rate for better scalability
            refresh_ms = self.gui_config.get('image_loader_refresh_ms', 50)
            self.msleep(refresh_ms)  # Check for new requests at configured interval
    
    def load_frame(self, frame_index: int):
        """
        Load and process images for specified frame with proper sizing.
        
        Args:
            frame_index: Index of frame to load
        """
        if frame_index >= len(self.frame_files):
            return
        
        frame_file = self.frame_files[frame_index]
        frame_stem = frame_file.stem  # e.g., "0", "1", "2"
        
        # Load images from all available cameras
        loaded_images = {}
        
        for camera_name, camera_path in self.image_paths.items():
            if not camera_path.exists():
                continue
            
            image_file = camera_path / f"{frame_stem}{frame_file.suffix}"
            if image_file.exists():
                img = cv2.imread(str(image_file))
                if img is not None:
                    # Apply proper sizing based on camera type and configuration
                    resized_img = self._resize_image(img, camera_name)
                    loaded_images[camera_name] = resized_img
        
        if not loaded_images:
            self.loading_error.emit(f"No images found for frame {frame_index}")
            return
        
        # Combine images for display
        combined_image = self._combine_images(loaded_images)
        
        # Create frame info string
        camera_list = ", ".join(loaded_images.keys())
        frame_info = f"Frame {frame_index + 1}/{len(self.frame_files)} | {frame_stem} | Cameras: {camera_list}"
        
        # Emit the loaded image and info
        self.images_loaded.emit(combined_image, frame_info)
    
    def _resize_image(self, image: np.ndarray, camera_name: str) -> np.ndarray:
        """
        Resize image according to configuration and camera type.
        
        Implements the sizing logic from the yaml file. Taking an example of:
        - Left/right cameras: max 854x480
        - Side cameras: max 640x480  
        - If resize_max_size true: always resize to max dimension
        - If false: only resize when input is larger than max size
        
        Args:
            image: Input image
            camera_name: Name of the camera (left, right, side_1, side_2)
            
        Returns:
            Resized image
        """
        # Determine max size based on camera type
        if camera_name in ['left', 'right']:
            max_size = self.main_image_max_size
        else:  # side cameras
            max_size = self.side_image_max_size
        
        h, w = image.shape[:2]
        
        # Safely extract max dimensions with defensive unpacking
        try:
            if isinstance(max_size, (list, tuple)) and len(max_size) >= 2:
                max_w, max_h = max_size[0], max_size[1]
            else:
                # Convert OmegaConf object or fallback to defaults
                max_size_list = OmegaConf.to_container(max_size) if hasattr(max_size, '_content') else list(max_size)
                max_w, max_h = max_size_list[0], max_size_list[1]
        except (ValueError, TypeError, IndexError) as e:
            # Error unpacking max_size - this should not happen with proper config
            error_msg = f"Error: Could not unpack max_size {max_size} for {camera_name}: {e}"
            print(error_msg)
            raise ValueError(f"Invalid image size configuration for {camera_name}. Please check config file.") from e
        
        # Check if resizing is needed
        needs_resize = False
        if self.resize_max_size:
            # Always resize to max dimension
            needs_resize = True
        else:
            # Only resize if larger than max size
            needs_resize = (w > max_w) or (h > max_h)
        
        if not needs_resize:
            return image
        
        # Calculate scale factor to fit within max size while maintaining aspect ratio
        # IMPORTANT: Use min() to ensure no cropping - image will fit entirely within bounds
        scale_w = max_w / w
        scale_h = max_h / h
        scale = min(scale_w, scale_h)  # This preserves aspect ratio and prevents cropping
        
        # Calculate new dimensions (proportional scaling only)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image proportionally - no content loss
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    def _combine_images(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine multiple camera images for display.
        
        Arranges images in a 2x2 grid layout:
        - Top row: Left and Right cameras (main stereo pair)
        - Bottom row: Side camera 1 and Side camera 2 (if available)
        - If side camera(s) is None, leave blank
        
        Args:
            images: Dictionary of camera images
            
        Returns:
            Combined image for display
        """
        if len(images) == 1:
            return list(images.values())[0]
        
        # Get individual camera images
        left_img = images.get('left')
        right_img = images.get('right')
        side1_img = images.get('side_1')
        side2_img = images.get('side_2')
        
        # If we only have left/right stereo pair, combine horizontally
        if left_img is not None and right_img is not None and side1_img is None and side2_img is None:
            stereo_images = self._resize_to_same_height([left_img, right_img])
            return cv2.hconcat(stereo_images)
        
        # For 2x2 grid layout, we need to create consistent cell sizes
        # Determine target dimensions for each quadrant
        all_images = [img for img in [left_img, right_img, side1_img, side2_img] if img is not None]
        if not all_images:
            return list(images.values())[0]
        
        # Calculate appropriate cell dimensions based on main image constraints
        # Main cameras get full resolution, side cameras get smaller resolution
        # Safely extract dimensions with defensive access
        try:
            if len(self.main_image_max_size) < 2 or len(self.side_image_max_size) < 2:
                raise ValueError("Image size config must have at least 2 dimensions [width, height]")
            
            main_height = int(self.main_image_max_size[1])
            main_width = int(self.main_image_max_size[0]) // 2  # Divide by 2 for side-by-side layout
            side_height = int(self.side_image_max_size[1])
            side_width = int(self.side_image_max_size[0])
        except (IndexError, TypeError, ValueError) as e:
            error_msg = f"Error accessing image size config: {e}"
            print(error_msg)
            raise ValueError("Invalid image size configuration - cannot create image layout") from e
        
        # Resize images directly to their max sizes using cv2.resize (no padding)
        # This works well for common input sizes: 1920x1080 -> 854x480, 640x480 -> 640x480
        
        if left_img is not None:
            # Resize to exactly main_width x main_height for grid consistency
            left_resized = cv2.resize(left_img, (main_width, main_height), interpolation=cv2.INTER_AREA)
        else:
            # Create blank placeholder
            left_resized = np.zeros((main_height, main_width, 3), dtype=np.uint8)
        
        if right_img is not None:
            # Resize to exactly main_width x main_height for grid consistency
            right_resized = cv2.resize(right_img, (main_width, main_height), interpolation=cv2.INTER_AREA)
        else:
            # Create blank placeholder  
            right_resized = np.zeros((main_height, main_width, 3), dtype=np.uint8)
        
        if side1_img is not None:
            # Resize to exactly side_width x side_height
            side1_resized = cv2.resize(side1_img, (side_width, side_height), interpolation=cv2.INTER_AREA)
        else:
            # Create blank placeholder
            side1_resized = np.zeros((side_height, side_width, 3), dtype=np.uint8)
        
        if side2_img is not None:
            # Resize to exactly side_width x side_height  
            side2_resized = cv2.resize(side2_img, (side_width, side_height), interpolation=cv2.INTER_AREA)
        else:
            # Create blank placeholder
            side2_resized = np.zeros((side_height, side_width, 3), dtype=np.uint8)
        
        # Create rows by concatenating images horizontally
        # All images in each row now have exact same dimensions, so no resizing needed
        
        # Top row: left and right main cameras  
        top_row = cv2.hconcat([left_resized, right_resized])
        
        # Bottom row: side cameras
        bottom_row = cv2.hconcat([side1_resized, side2_resized])
        
        # Ensure rows have same width for proper grid alignment
        if top_row.shape[1] != bottom_row.shape[1]:
            target_width = max(top_row.shape[1], bottom_row.shape[1])
            
            # Use cv2.resize to match widths ( avoiding padding)
            if top_row.shape[1] < target_width:
                top_row = cv2.resize(top_row, (target_width, top_row.shape[0]), interpolation=cv2.INTER_AREA)
            if bottom_row.shape[1] < target_width:
                bottom_row = cv2.resize(bottom_row, (target_width, bottom_row.shape[0]), interpolation=cv2.INTER_AREA)
        
        # Combine top and bottom rows vertically
        combined = cv2.vconcat([top_row, bottom_row])
        return combined
    
    def _resize_to_fit(self, image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        Resize image to fit within target dimensions while maintaining aspect ratio.
        
        Args:
            image: Input image
            target_width: Maximum width
            target_height: Maximum height
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        # Calculate scale factor to fit within target dimensions
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        # Calculate new dimensions using precise rounding to avoid edge cutting
        new_w = round(w * scale)
        new_h = round(h * scale)
        
        # Ensure minimum dimensions
        new_w = max(new_w, 1)
        new_h = max(new_h, 1)
        
        # Resize image
        if scale < 1.0:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = image.copy()
        
        # Pad to exact target dimensions if needed (center the image)
        if new_w < target_width or new_h < target_height:
            # Create target-sized canvas
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate position to center the image
            start_y = (target_height - new_h) // 2
            start_x = (target_width - new_w) // 2
            
            # Place resized image on canvas
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            return canvas
        
        return resized
    
    def _pad_to_width(self, image: np.ndarray, target_width: int) -> np.ndarray:
        """
        Pad image to target width by adding black borders on the sides.
        This preserves the original image content without any distortion.
        
        Args:
            image: Input image
            target_width: Desired width
            
        Returns:
            Padded image with exact target width
        """
        h, w = image.shape[:2]
        if w >= target_width:
            return image
        
        # Calculate padding needed
        pad_total = target_width - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        # Create padded image with black borders
        if len(image.shape) == 3:
            padded = np.zeros((h, target_width, image.shape[2]), dtype=image.dtype)
            padded[:, pad_left:pad_left+w, :] = image
        else:
            padded = np.zeros((h, target_width), dtype=image.dtype)
            padded[:, pad_left:pad_left+w] = image
        
        return padded
    
    def _resize_to_same_height(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Resize images to have the same height."""
        if not images:
            return images
        
        target_height = max(img.shape[0] for img in images)
        resized = []
        
        for img in images:
            if img.shape[0] != target_height:
                scale = target_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
            resized.append(img)
        
        return resized
    
    def stop(self):
        """Stop the thread."""
        self.running = False


class DataAnnotationGUI(QMainWindow):
    """
    Main GUI class for data annotation.
    
    Provides a comprehensive interface for annotating surgical events, phases,
    and contact detection in dVRK data. Implements all features
    
    - Multi-camera video display with proper sizing
    - Event, phase, and contact annotation
    - Multi-frame batch labeling
    - Auto-play functionality with configurable speed (1x = 30 Hz)
    - Proper folder structure for saving annotations
    - PSM contact detection with boolean labels
    - No auto-save, no default loading
    """
    
    def __init__(self):
        super().__init__()
        
        # Core components
        self.config_loader = ConfigLoader()
        self.config: Optional[DictConfig] = None
        self.image_processor: Optional[ImageProcessor] = None
        
        # Data management
        self.image_paths: Dict[str, Path] = {}
        self.frame_files: List[Path] = []
        self.current_frame_index = 0
        
        # Annotation data storage
        self.annotations: Dict[int, List[Dict]] = {}  # frame_index -> list of annotations
        
        # Save folder management
        self.custom_save_folder: Optional[Path] = None
        
        # PSM configuration - can present maximum 3 PSMs
        self.available_psms = ['PSM1', 'PSM2', 'PSM3']
        self.active_psms = []  # Will be populated from config file
        
        # Multi-frame labeling support
        self.multi_frame_mode = False
        self.frame_range_start: Optional[int] = None
        self.frame_range_end: Optional[int] = None
        
        # Auto-play functionality (1x = 30 Hz as specified)
        self.auto_play_timer = QTimer()
        self.auto_play_timer.timeout.connect(self._auto_play_step)
        self.is_playing = False
        self.playback_speed_ms = 33  # ~30 Hz (1000ms / 30 = 33.33ms)
        self.speed_multiplier = 1.0  # 1x speed
        
        # GUI components (initialized in setup methods)
        self.video_display_label: Optional[QLabel] = None
        self.timeline_slider: Optional[QSlider] = None
        self.frame_info_label: Optional[QLabel] = None
        self.annotation_list: Optional[QListWidget] = None
        
        # Event annotation controls
        self.event_input: Optional[QLineEdit] = None
        
        # Phase annotation controls
        self.phase_inputs: Dict[str, QLineEdit] = {}
        
        # Contact annotation controls  
        self.contact_checkboxes: Dict[str, QCheckBox] = {}
        
        # Multi-frame controls
        self.multi_frame_checkbox: Optional[QCheckBox] = None
        self.range_display_label: Optional[QLabel] = None
        
        # Camera button controls for opening original images
        self.camera_buttons: Dict[str, QPushButton] = {}
        self.left_camera_btn: Optional[QPushButton] = None
        self.right_camera_btn: Optional[QPushButton] = None
        self.side1_camera_btn: Optional[QPushButton] = None
        self.side2_camera_btn: Optional[QPushButton] = None
        
        # Independent image windows tracking (for separate, non-modal windows)
        self._image_windows: List = []
        
        
        # Initialize the GUI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface components."""
        self.setWindowTitle("dVRK Data Annotation Tool")
        
        # CRITICAL FIX: GUI optimized for 1080P with FULL visibility (no parts cut off)
        # Get GUI configuration parameters - using configurable values for better scalability
        gui_config = self.config_loader.get_gui_config() if hasattr(self, 'config_loader') else {
            'window_width': 1910, 'window_height': 1000
        }

        # Set window geometry using configurable parameters
        # Position at (5, 10) to minimize top edge while ensuring full bottom visibility
        # Height set to 1000px (safe maximum for 1080P with taskbar + decorations)
        # This ensures NO parts of GUI are cut off at bottom
        self.setGeometry(5, 10, gui_config['window_width'], gui_config['window_height'])
        
        # Set application style
        self.setStyleSheet(self._get_app_stylesheet())
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create main splitter to divide video and controls
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left side: Video display area (larger for better left/right viewing)
        video_widget = self._create_video_widget()
        main_splitter.addWidget(video_widget)
        
        # Right side: Control panel
        control_widget = self._create_control_widget()
        main_splitter.addWidget(control_widget)
        
        # Set splitter proportions optimized for 1920x1080 - video gets more space
        main_splitter.setSizes([1300, 500])
        
        # Create status bar and menu
        self.statusBar().showMessage("Ready - Load configuration to begin")
        self._create_menu_bar()
    
    def _get_app_stylesheet(self) -> str:
        """Get the application stylesheet."""
        return """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ccc;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                font-size: 14px;
                border-radius: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #4CAF50;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """
    
    def _create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_config_action = file_menu.addAction('Load Configuration')
        load_config_action.triggered.connect(self.load_configuration)
        
        file_menu.addSeparator()
        
        save_action = file_menu.addAction('Save Annotations')
        save_action.triggered.connect(self.save_annotations)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
        
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)
    
    def _create_video_widget(self) -> QWidget:
        """Create the video display widget."""
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        # Video display group
        video_group = QGroupBox("Video Display")
        video_group_layout = QVBoxLayout(video_group)
        
        # Video display label - size will be set from config after loading
        self.video_display_label = QLabel()
        self.video_display_label.setAlignment(Qt.AlignCenter)
        # Initial minimum size, will be updated from config
        self.video_display_label.setMinimumSize(800, 600)
        self.video_display_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                background-color: #000;
                color: white;
                font-size: 16px;
            }
        """)
        self.video_display_label.setText("Load configuration to display video")
        
        # Wrap in scroll area for large images
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.video_display_label)
        scroll_area.setWidgetResizable(True)
        video_group_layout.addWidget(scroll_area)
        
        # Original image buttons (arranged to match the 2x2 camera layout)
        camera_buttons_widget = self._create_camera_buttons_widget()
        video_group_layout.addWidget(camera_buttons_widget)
        
        # Timeline controls
        timeline_widget = self._create_timeline_widget()
        video_group_layout.addWidget(timeline_widget)
        
        video_layout.addWidget(video_group)
        return video_widget
    
    def _create_timeline_widget(self) -> QWidget:
        """Create timeline control widget with auto-play functionality."""
        timeline_widget = QWidget()
        timeline_layout = QVBoxLayout(timeline_widget)
        
        # Frame information label
        self.frame_info_label = QLabel("No frames loaded")
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        self.frame_info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        timeline_layout.addWidget(self.frame_info_label)
        
        # Timeline slider
        slider_layout = QHBoxLayout()
        
        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(self.previous_frame)
        slider_layout.addWidget(prev_btn)
        
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.valueChanged.connect(self._on_timeline_changed)
        self.timeline_slider.setEnabled(False)
        slider_layout.addWidget(self.timeline_slider)
        
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self.next_frame)
        slider_layout.addWidget(next_btn)
        
        timeline_layout.addLayout(slider_layout)

        # IMPROVEMENT: Visual annotation progress indicator
        # Shows which frames have been annotated vs unannotated
        annotation_indicator_layout = QHBoxLayout()
        annotation_indicator_layout.addWidget(QLabel("Annotation Status:"))

        self.annotation_progress_label = QLabel("No data loaded")
        self.annotation_progress_label.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #2196F3; padding: 3px;"
        )
        annotation_indicator_layout.addWidget(self.annotation_progress_label)

        # Add a progress bar showing annotation completion
        self.annotation_progress_bar = QProgressBar()
        self.annotation_progress_bar.setMaximum(100)
        self.annotation_progress_bar.setValue(0)
        self.annotation_progress_bar.setTextVisible(True)
        self.annotation_progress_bar.setFormat("%p% annotated")
        self.annotation_progress_bar.setMaximumHeight(15)
        self.annotation_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        annotation_indicator_layout.addWidget(self.annotation_progress_bar)

        # Current frame annotation status indicator
        self.current_frame_status_label = QLabel("⚪")
        self.current_frame_status_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 3px;"
        )
        self.current_frame_status_label.setToolTip("Current frame annotation status")
        annotation_indicator_layout.addWidget(self.current_frame_status_label)

        timeline_layout.addLayout(annotation_indicator_layout)

        # Auto-play controls with speed configuration
        playback_layout = QHBoxLayout()
        
        # Jump to start/end - start and end buttons together (end button moved to right of start )
        jump_start_btn = QPushButton("⏮ Start")
        jump_start_btn.clicked.connect(lambda: self.jump_to_frame(0))
        playback_layout.addWidget(jump_start_btn)
        
        jump_end_btn = QPushButton("End ⏭")
        jump_end_btn.clicked.connect(lambda: self.jump_to_frame(len(self.frame_files) - 1 if self.frame_files else 0))
        playback_layout.addWidget(jump_end_btn)
        
        # Play/pause button
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.clicked.connect(self.toggle_auto_play)
        playback_layout.addWidget(self.play_pause_btn)
        
        # Speed control with dropdown menu (0.1x, 0.25x, 0.5x, 1x [30.00 Hz], 2x, 4x)
        playback_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        speed_options = [
            ("0.1x", 333),   # 333ms = ~3 Hz (0.1x of 30 Hz)
            ("0.25x", 133),  # 133ms = ~7.5 Hz (0.25x of 30 Hz)
            ("0.5x", 67),    # 67ms = ~15 Hz (0.5x of 30 Hz)
            ("1x [30.00 Hz]", 33),  # 33ms = ~30 Hz (1x baseline)
            ("2x", 17),      # 17ms = ~60 Hz (2x of 30 Hz)
            ("4x", 8),       # 8ms = ~125 Hz (4x of 30 Hz)
        ]
        
        for speed_label, speed_ms in speed_options:
            self.speed_combo.addItem(speed_label, speed_ms)
        
        # Set default to 1x speed
        self.speed_combo.setCurrentText("1x [30.00 Hz]")
        self.speed_combo.currentTextChanged.connect(self._on_speed_combo_changed)
        self.speed_combo.setMaximumWidth(120)
        playback_layout.addWidget(self.speed_combo)
        
        # Frequency and frame display for auto-play status
        self.playback_status_label = QLabel("Ready")
        self.playback_status_label.setMinimumWidth(120)
        self.playback_status_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        playback_layout.addWidget(self.playback_status_label)
        
        # Frame jump input
        playback_layout.addWidget(QLabel("Jump to:"))
        self.frame_jump_input = QSpinBox()
        self.frame_jump_input.setMinimum(1)
        self.frame_jump_input.setMaximum(1)
        self.frame_jump_input.valueChanged.connect(lambda v: self.jump_to_frame(v - 1))
        playback_layout.addWidget(self.frame_jump_input)
        
        timeline_layout.addLayout(playback_layout)
        
        return timeline_widget
    
    def _create_camera_buttons_widget(self) -> QWidget:
        """Create buttons for opening original images in new windows."""
        buttons_widget = QWidget()
        buttons_layout = QVBoxLayout(buttons_widget)
        
        # Add a descriptive label
        info_label = QLabel("Open Original Images:")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-weight: bold; color: #666;")
        buttons_layout.addWidget(info_label)
        
        # Create 2x2 grid layout to match camera arrangement
        grid_layout = QGridLayout()
        
        # Top row: Left and Right cameras
        self.left_camera_btn = QPushButton("Left Camera")
        self.left_camera_btn.clicked.connect(lambda: self._open_original_image('left'))
        self.left_camera_btn.setEnabled(False)  # Initially disabled until config is loaded
        grid_layout.addWidget(self.left_camera_btn, 0, 0)
        
        self.right_camera_btn = QPushButton("Right Camera")
        self.right_camera_btn.clicked.connect(lambda: self._open_original_image('right'))
        self.right_camera_btn.setEnabled(False)
        grid_layout.addWidget(self.right_camera_btn, 0, 1)
        
        # Bottom row: Side cameras
        self.side1_camera_btn = QPushButton("Side Camera 1")
        self.side1_camera_btn.clicked.connect(lambda: self._open_original_image('side_1'))
        self.side1_camera_btn.setEnabled(False)
        grid_layout.addWidget(self.side1_camera_btn, 1, 0)
        
        self.side2_camera_btn = QPushButton("Side Camera 2")
        self.side2_camera_btn.clicked.connect(lambda: self._open_original_image('side_2'))
        self.side2_camera_btn.setEnabled(False)
        grid_layout.addWidget(self.side2_camera_btn, 1, 1)
        
        buttons_layout.addLayout(grid_layout)
        
        # Store button references for later enabling/disabling based on available cameras
        self.camera_buttons = {
            'left': self.left_camera_btn,
            'right': self.right_camera_btn,
            'side_1': self.side1_camera_btn,
            'side_2': self.side2_camera_btn
        }
        
        return buttons_widget
    
    def _create_control_widget(self) -> QWidget:
        """Create the control panel widget with reorganized two-column layout."""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # Multi-frame labeling controls span both columns
        multi_frame_group = self._create_multi_frame_section()
        control_layout.addWidget(multi_frame_group)
        
        # Create two-column layout for the main controls
        columns_layout = QHBoxLayout()
        
        # Left column: Configuration, Quick Actions, Event, Phase (when loaded), Contact (when loaded)
        left_column = QVBoxLayout()
        left_column_widget = QWidget()
        left_column_widget.setLayout(left_column)
        
        config_group = self._create_config_section()
        left_column.addWidget(config_group)
        
        quick_actions_group = self._create_quick_actions_section()
        left_column.addWidget(quick_actions_group)
        
        event_group = self._create_event_section()
        left_column.addWidget(event_group)
        
        # Phase and contact sections will be added dynamically after config load
        # Store references for dynamic insertion
        self.left_column_layout = left_column
        
        # Add stretch to push content to top of left column
        left_column.addStretch()
        
        # Right column: Current annotations, statistics, save
        right_column = QVBoxLayout()
        right_column_widget = QWidget()
        right_column_widget.setLayout(right_column)
        
        current_group = self._create_current_annotations_section()
        right_column.addWidget(current_group)
        
        stats_group = self._create_statistics_section()
        right_column.addWidget(stats_group)
        
        save_group = self._create_save_section()
        right_column.addWidget(save_group)
        
        # Add stretch to push content to top of right column
        right_column.addStretch()
        
        # Add columns to horizontal layout
        columns_layout.addWidget(left_column_widget)
        columns_layout.addWidget(right_column_widget)
        
        # Set equal spacing for columns
        columns_layout.setSpacing(10)
        
        # Add columns layout to main control layout
        control_layout.addLayout(columns_layout)
        
        return control_widget
    
    def _create_config_section(self) -> QGroupBox:
        """Create configuration section."""
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)
        
        self.config_status_label = QLabel("No configuration loaded")
        config_layout.addWidget(self.config_status_label)
        
        load_config_btn = QPushButton("Load Configuration")
        load_config_btn.clicked.connect(self.load_configuration)
        config_layout.addWidget(load_config_btn)
        
        # Annotation loading controls
        load_annotations_btn = QPushButton("Load Existing Annotations")
        load_annotations_btn.clicked.connect(self.load_existing_annotations_dialog)
        config_layout.addWidget(load_annotations_btn)
        
        
        return config_group
    
    def _create_multi_frame_section(self) -> QGroupBox:
        """Create multi-frame labeling section."""
        multi_frame_group = QGroupBox("Multi-Frame Labeling")
        multi_frame_layout = QVBoxLayout(multi_frame_group)
        
        # Enable multi-frame mode
        self.multi_frame_checkbox = QCheckBox("Enable multi-frame labeling")
        self.multi_frame_checkbox.toggled.connect(self._on_multi_frame_toggled)
        multi_frame_layout.addWidget(self.multi_frame_checkbox)
        
        # Range selection controls
        range_layout = QHBoxLayout()
        
        self.set_start_btn = QPushButton("Set Start")
        self.set_start_btn.clicked.connect(self._set_range_start)
        self.set_start_btn.setEnabled(False)
        range_layout.addWidget(self.set_start_btn)
        
        self.set_end_btn = QPushButton("Set End")
        self.set_end_btn.clicked.connect(self._set_range_end)
        self.set_end_btn.setEnabled(False)
        range_layout.addWidget(self.set_end_btn)
        
        multi_frame_layout.addLayout(range_layout)
        
        # Range display
        self.range_display_label = QLabel("Range: Not set")
        self.range_display_label.setStyleSheet("font-style: italic;")
        multi_frame_layout.addWidget(self.range_display_label)
        
        return multi_frame_group
    
    def _create_event_section(self) -> QGroupBox:
        """
        Create enhanced event annotation section supporting multiple events.

        IMPROVEMENT: Supports multiple events per frame with sequencing:
        - Add multiple events to the same frame
        - Reorder events (Move Up/Down)
        - Remove individual events
        - Shows current events for the frame in a list
        """
        event_group = QGroupBox("Event Annotation (Multi-Event Support)")
        event_layout = QVBoxLayout(event_group)

        # Input for new event
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Event:"))
        self.event_input = QLineEdit()
        self.event_input.setPlaceholderText("Enter event name...")
        self.event_input.returnPressed.connect(self._add_event_annotation)  # Add on Enter
        input_layout.addWidget(self.event_input)
        event_layout.addLayout(input_layout)

        # Add event button
        add_event_btn = QPushButton("+ Add Event")
        add_event_btn.clicked.connect(self._add_event_annotation)
        add_event_btn.setStyleSheet("background-color: #4CAF50; font-weight: bold;")
        event_layout.addWidget(add_event_btn)

        # IMPROVEMENT: List showing current frame's events with ordering
        event_layout.addWidget(QLabel("Current Frame Events:"))
        self.current_frame_events_list = QListWidget()
        self.current_frame_events_list.setMaximumHeight(80)
        self.current_frame_events_list.setStyleSheet(
            "QListWidget { background-color: #f9f9f9; border: 1px solid #ccc; }"
        )
        event_layout.addWidget(self.current_frame_events_list)

        # IMPROVEMENT: Event management buttons (Remove, Move Up/Down)
        event_mgmt_layout = QHBoxLayout()

        remove_event_btn = QPushButton("− Remove")
        remove_event_btn.clicked.connect(self._remove_selected_event)
        remove_event_btn.setStyleSheet("background-color: #f44336; color: white;")
        event_mgmt_layout.addWidget(remove_event_btn)

        move_up_btn = QPushButton("↑ Move Up")
        move_up_btn.clicked.connect(self._move_event_up)
        move_up_btn.setStyleSheet("background-color: #2196F3; color: white;")
        event_mgmt_layout.addWidget(move_up_btn)

        move_down_btn = QPushButton("↓ Move Down")
        move_down_btn.clicked.connect(self._move_event_down)
        move_down_btn.setStyleSheet("background-color: #2196F3; color: white;")
        event_mgmt_layout.addWidget(move_down_btn)

        event_layout.addLayout(event_mgmt_layout)

        return event_group
    
    def _create_phase_section(self) -> QGroupBox:
        """Create phase annotation section with text inputs and PSM selection."""
        phase_group = QGroupBox("Phase Annotation")
        phase_layout = QFormLayout(phase_group)
        
        # Create text inputs for each active PSM
        self.phase_inputs = {}
        
        for psm in self.active_psms:
            phase_input = QLineEdit()
            phase_input.setPlaceholderText(f"Enter phase for {psm}...")
            self.phase_inputs[psm] = phase_input
            phase_layout.addRow(f"{psm} Phase:", phase_input)
        
        add_phase_btn = QPushButton("Add Phase")
        add_phase_btn.clicked.connect(self._add_phase_annotation)
        phase_layout.addWidget(add_phase_btn)
        
        return phase_group
    
    def _create_contact_section(self) -> QGroupBox:
        """
        Create contact annotation section.

        FIXED: Compact layout to fit 3 PSMs without requiring scroll.
        Uses horizontal layout for checkboxes to save vertical space.

        FIXED: Increased font sizes from 10px to 12px for better readability.
        """
        contact_group = QGroupBox("Contact Detection")
        contact_layout = QVBoxLayout(contact_group)  # Changed from QFormLayout for better control

        # Label with readable font size
        label = QLabel("Select PSMs with contact:")
        label.setStyleSheet("font-size: 12px; margin-bottom: 2px;")  # FIXED: 10px → 12px
        contact_layout.addWidget(label)

        # FIXED: Use horizontal layout for checkboxes to save vertical space
        # This allows all 3 PSMs to fit without scrolling
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setSpacing(8)  # Slightly more spacing for readability

        # Create checkboxes for each active PSM with readable font
        for psm in self.active_psms:
            checkbox = QCheckBox(psm)  # Just PSM name (compact)
            checkbox.setStyleSheet("font-size: 12px;")  # FIXED: 10px → 12px for readability
            self.contact_checkboxes[psm] = checkbox
            checkbox_layout.addWidget(checkbox)

        checkbox_layout.addStretch()  # Push checkboxes to left
        contact_layout.addLayout(checkbox_layout)

        # Button with readable font
        add_contact_btn = QPushButton("Add Contact State")
        add_contact_btn.setStyleSheet("font-size: 12px; padding: 5px;")  # FIXED: 10px → 12px, padding 3px → 5px
        add_contact_btn.clicked.connect(self._add_contact_annotation)
        contact_layout.addWidget(add_contact_btn)

        # Minimize vertical spacing while maintaining readability
        contact_layout.setSpacing(5)  # FIXED: 3px → 5px for better visual separation
        contact_layout.setContentsMargins(5, 5, 5, 5)

        return contact_group
    
    def _create_current_annotations_section(self) -> QGroupBox:
        """Create current frame annotations display section."""
        current_group = QGroupBox("Current Frame Annotations")
        current_layout = QVBoxLayout(current_group)
        
        self.annotation_list = QListWidget()
        # Use configurable annotation list height for better scalability
        gui_config = self.config_loader.get_gui_config() if hasattr(self, 'config_loader') else {'annotation_list_height': 150}
        self.annotation_list.setMaximumHeight(gui_config['annotation_list_height'])
        self.annotation_list.itemDoubleClicked.connect(self._edit_annotation)
        current_layout.addWidget(self.annotation_list)
        
        # Annotation management buttons
        annotation_btn_layout = QHBoxLayout()
        
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._remove_selected_annotation)
        annotation_btn_layout.addWidget(self.remove_btn)
        
        self.clear_btn = QPushButton("Clear Frame")
        self.clear_btn.clicked.connect(self._clear_frame_annotations)
        annotation_btn_layout.addWidget(self.clear_btn)
        
        # Add "Clear All" button to clear all annotations from all frames
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self._clear_all_annotations)
        self.clear_all_btn.setStyleSheet("background-color: #ff6b6b; color: white; font-weight: bold;")
        annotation_btn_layout.addWidget(self.clear_all_btn)
        
        current_layout.addLayout(annotation_btn_layout)
        
        return current_group
    
    def _create_statistics_section(self) -> QGroupBox:
        """Create statistics section with reset/reload button."""
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("No data loaded")
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 11px; padding: 5px;")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        # Add reset/reload button for statistics
        reset_stats_btn = QPushButton("Reset/Reload Stats")
        reset_stats_btn.clicked.connect(self._reset_statistics)
        reset_stats_btn.setStyleSheet("font-size: 12px; padding: 5px;")
        stats_layout.addWidget(reset_stats_btn)
        
        return stats_group
    
    def _create_quick_actions_section(self) -> QGroupBox:
        """
        Create quick actions section.
        
        As specified in additional comments: Quick actions only include 
        the quick annotation of contact for PSMs.
        """
        quick_group = QGroupBox("Quick Actions - Contact")
        quick_layout = QVBoxLayout(quick_group)
        
        # Quick contact annotation buttons for PSMs only
        quick_buttons = [
            ("All PSMs Contact", lambda: self._quick_contact_all(True)),
            ("No PSM Contact", lambda: self._quick_contact_all(False))
        ]
        
        # Add individual PSM quick contact buttons
        for psm in self.active_psms:
            quick_buttons.append(
                (f"{psm} Contact Only", lambda p=psm: self._quick_contact_single(p, True))
            )
        
        for label, callback in quick_buttons:
            btn = QPushButton(label)
            btn.clicked.connect(callback)
            quick_layout.addWidget(btn)
        
        return quick_group
    
    def _create_save_section(self) -> QGroupBox:
        """Create save section with folder selection."""
        save_group = QGroupBox("Save")
        save_layout = QVBoxLayout(save_group)
        
        # Display current save folder
        self.save_folder_label = QLabel("No folder selected")
        self.save_folder_label.setWordWrap(True)
        self.save_folder_label.setStyleSheet("font-size: 10px; color: #666; padding: 5px;")
        save_layout.addWidget(self.save_folder_label)
        
        # Button to select save folder
        select_folder_btn = QPushButton("Select Save Folder")
        select_folder_btn.clicked.connect(self.select_save_folder)
        save_layout.addWidget(select_folder_btn)
        
        # Button to save annotations to selected folder
        save_btn = QPushButton("Save Annotations")
        save_btn.clicked.connect(self.save_annotations)
        save_layout.addWidget(save_btn)
        
        return save_group
    
    def _recreate_psm_dependent_sections(self):
        """
        Recreate phase and contact sections when PSM configuration changes.
        
        This method is called when loading configuration to update the UI
        based on which PSMs are enabled in the config file.
        """
        if hasattr(self, 'left_column_layout'):
            # Remove existing phase and contact sections from left column
            for i in reversed(range(self.left_column_layout.count())):
                item = self.left_column_layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    if isinstance(widget, QGroupBox):
                        if widget.title() in ["Phase Annotation", "Contact Detection"]:
                            self.left_column_layout.removeWidget(widget)
                            widget.deleteLater()
            
            # Insert phase and contact sections after event section (before stretch)
            # Find the stretch item and insert before it
            stretch_index = self.left_column_layout.count() - 1  # Last item should be stretch
            
            # Create and insert new phase section
            phase_group = self._create_phase_section()
            self.left_column_layout.insertWidget(stretch_index, phase_group)
            
            # Create and insert new contact section
            contact_group = self._create_contact_section()
            self.left_column_layout.insertWidget(stretch_index + 1, contact_group)
    
    
    def load_configuration(self):
        """Load configuration and initialize data paths."""
        try:
            # Load configuration
            config_file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Configuration File",
                str(proj_root / "config"),
                "YAML files (*.yaml *.yml)"
            )
            
            if not config_file:
                return
            
            self.config = self.config_loader.load_config(Path(config_file))
            
            # Get image paths from configuration
            self.image_paths = self.config_loader.get_image_paths()
            
            # Get PSM configuration from config file
            psm_config = self.config_loader.get_psm_config()
            self.active_psms = [psm for psm, enabled in psm_config.items() if enabled]
            print(f"Active PSMs from config: {self.active_psms}")
            
            # Recreate UI sections with updated PSM configuration
            self._recreate_psm_dependent_sections()
            
            
            # Verify paths exist and get frame files
            existing_cameras = []
            for camera_name, camera_path in self.image_paths.items():
                if camera_path.exists():
                    existing_cameras.append(camera_name)
            
            if not existing_cameras:
                raise FileNotFoundError("No camera image directories found")
            
            # Enable camera buttons for available cameras
            self._update_camera_buttons(existing_cameras)
            
            # Load frame list (use first available camera as reference)
            reference_camera = existing_cameras[0]
            self.frame_files = glob_sorted_frame(self.image_paths[reference_camera])
            
            if not self.frame_files:
                raise ValueError("No image files found")
            
            # Initialize timeline
            self.timeline_slider.setMaximum(len(self.frame_files) - 1)
            self.timeline_slider.setEnabled(True)
            self.frame_jump_input.setMaximum(len(self.frame_files))
            self.frame_jump_input.setValue(1)
            
            # Start image processor thread
            if self.image_processor:
                self.image_processor.stop()
                self.image_processor.wait()
            
            size_config = self.config_loader.get_image_size_config()
            gui_config = self.config_loader.get_gui_config()
            self.image_processor = ImageProcessor(self.image_paths, self.frame_files, size_config, gui_config)
            self.image_processor.images_loaded.connect(self._on_images_loaded)
            self.image_processor.loading_error.connect(self._on_loading_error)
            self.image_processor.start()
            
            # Load first frame
            self.jump_to_frame(0)
            
            # Update status
            dataset_info = f"Config: {Path(config_file).stem}"
            self.config_status_label.setText(f"Loaded: {dataset_info}")
            self.statusBar().showMessage(
                f"Configuration loaded - {len(self.frame_files)} frames, cameras: {', '.join(existing_cameras)}"
            )
            
            # Initialize statistics display
            self._update_statistics()
            
            # Update save folder display if no custom folder is set
            if not self.custom_save_folder:
                config_save_folder = self.config_loader.get_save_folder()
                folder_str = str(config_save_folder)
                if len(folder_str) > 50:
                    folder_str = "..." + folder_str[-47:]
                self.save_folder_label.setText(f"Config: {folder_str}")
            
            # Update video display size based on config max sizes
            self._update_video_display_size()
            
        except Exception as e:
            error_msg = f"Error loading configuration: {str(e)}"
            print(f"Configuration error: {traceback.format_exc()}")
            QMessageBox.critical(self, "Configuration Error", error_msg)
            self.statusBar().showMessage("Configuration load failed")
    
    @pyqtSlot(np.ndarray, str)
    def _on_images_loaded(self, combined_image: np.ndarray, frame_info: str):
        """Handle loaded images from background thread."""
        try:
            # Convert OpenCV image to Qt format
            height, width, channel = combined_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(combined_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Create pixmap and display
            pixmap = QPixmap.fromImage(q_image)
            self.video_display_label.setPixmap(pixmap)
            
            # Update frame info
            self.frame_info_label.setText(frame_info)
            
            # Update annotation list for current frame
            self._update_annotation_list()
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    @pyqtSlot(str)
    def _on_loading_error(self, error_message: str):
        """Handle image loading errors."""
        print(f"Image loading error: {error_message}")
        self.statusBar().showMessage(f"Loading error: {error_message}")
    
    def _on_timeline_changed(self, value: int):
        """Handle timeline slider changes."""
        if self.image_processor:
            self.current_frame_index = value
            self.image_processor.set_frame_index(value)
            self.frame_jump_input.setValue(value + 1)
    
    def jump_to_frame(self, frame_index: int):
        """Jump to specific frame."""
        if 0 <= frame_index < len(self.frame_files):
            self.timeline_slider.setValue(frame_index)
    
    def previous_frame(self):
        """Go to previous frame."""
        if self.current_frame_index > 0:
            self.jump_to_frame(self.current_frame_index - 1)
    
    def next_frame(self):
        """Go to next frame."""
        if self.current_frame_index < len(self.frame_files) - 1:
            self.jump_to_frame(self.current_frame_index + 1)
    
    def _open_original_image(self, camera_name: str):
        """
        Open the original (unresized) image for the specified camera in a new window.
        
        Args:
            camera_name: Name of the camera ('left', 'right', 'side_1', 'side_2')
        """
        try:
            # Check if we have a valid configuration and current frame
            if not self.config or not self.frame_files:
                QMessageBox.warning(self, "No Data", "Please load configuration and data first.")
                return
            
            # Check if this camera path exists
            if camera_name not in self.image_paths:
                QMessageBox.warning(self, "Camera Not Available", 
                                  f"{camera_name.replace('_', ' ').title()} camera is not available for this dataset.")
                return
            
            # Get the current frame's image path for this camera
            frame_filename = self.frame_files[self.current_frame_index].name
            camera_image_path = self.image_paths[camera_name] / frame_filename
            
            if not camera_image_path.exists():
                QMessageBox.warning(self, "Image Not Found", 
                                  f"Original image not found:\n{camera_image_path}")
                return
            
            # Load the original image without any resizing
            import cv2
            original_image = cv2.imread(str(camera_image_path))
            if original_image is None:
                QMessageBox.critical(self, "Loading Error", 
                                   f"Failed to load image:\n{camera_image_path}")
                return
            
            # Convert BGR to RGB for Qt display
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Create a new window to display the original image
            self._create_original_image_window(original_image_rgb, camera_name, frame_filename)
            
        except Exception as e:
            error_msg = f"Error opening original image for {camera_name}: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "Image Display Error", error_msg)
    
    def _create_original_image_window(self, image: np.ndarray, camera_name: str, frame_filename: str):
        """
        Create a separate, independent window to display the original image.
        
        Window features:
        - Non-modal (independent from main window)
        - Resizable, maximizable, minimizable
        - Multiple windows can be open simultaneously
        - Proper window management with taskbar integration
        
        Args:
            image: Original image in RGB format
            camera_name: Name of the camera
            frame_filename: Filename of the current frame
        """
        from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QScrollArea, QWidget
        from PyQt5.QtGui import QPixmap, QImage
        from PyQt5.QtCore import Qt
        
        # Create a separate, independent main window (not a modal dialog)
        window = QMainWindow()
        window.setWindowTitle(f"Original {camera_name.replace('_', ' ').title()} - Frame {self.current_frame_index + 1} ({frame_filename})")
        
        # Enable all standard window controls (minimize, maximize, close)
        window.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        # Set reasonable initial window size but allow full resizing
        height, width, channel = image.shape
        initial_width = min(width + 50, 1200)
        initial_height = min(height + 150, 960)  # Extra space for info bar
        window.resize(initial_width, initial_height)
        
        # Create central widget and layout
        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add information label at the top
        info_text = f"Camera: {camera_name.replace('_', ' ').title()}  |  "
        info_text += f"Frame: {self.current_frame_index + 1} / {len(self.frame_files)}  |  "
        info_text += f"File: {frame_filename}  |  "
        info_text += f"Original Size: {width} x {height}"
        
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("""
            QLabel {
                font-family: monospace; 
                background-color: #f5f5f5; 
                padding: 8px; 
                border: 1px solid #ccc; 
                font-weight: bold;
                color: #333;
            }
        """)
        layout.addWidget(info_label)
        
        # Create image label
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(200, 200)  # Minimum size to prevent window from becoming too small
        
        # Convert numpy array to QPixmap
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        image_label.setPixmap(pixmap)
        
        # Wrap in scroll area for very large images
        scroll_area = QScrollArea()
        scroll_area.setWidget(image_label)
        scroll_area.setWidgetResizable(False)  # Don't resize the widget, allow scrolling instead
        scroll_area.setAlignment(Qt.AlignCenter)
        layout.addWidget(scroll_area)
        
        # Store reference to prevent garbage collection and enable independent operation
        if not hasattr(self, '_image_windows'):
            self._image_windows = []
        self._image_windows.append(window)
        
        # Clean up closed windows from the list
        def on_window_destroyed():
            if window in self._image_windows:
                self._image_windows.remove(window)
        
        window.destroyed.connect(on_window_destroyed)
        
        # Show the window as independent (non-modal)
        window.show()
        window.raise_()  # Bring to front
        window.activateWindow()  # Give it focus
    
    def _update_camera_buttons(self, existing_cameras: List[str]):
        """
        Enable/disable camera buttons based on available cameras.
        
        Args:
            existing_cameras: List of camera names that have valid image paths
        """
        # Enable buttons for available cameras, disable for unavailable ones
        for camera_name, button in self.camera_buttons.items():
            if camera_name in existing_cameras:
                button.setEnabled(True)
                button.setToolTip(f"Open original {camera_name.replace('_', ' ')} camera image")
            else:
                button.setEnabled(False) 
                button.setToolTip(f"{camera_name.replace('_', ' ').title()} camera not available for this dataset")
        
        # Update status message
        enabled_cameras = [name.replace('_', ' ').title() for name in existing_cameras]
        self.statusBar().showMessage(f"Available cameras: {', '.join(enabled_cameras)}")
    
    def toggle_auto_play(self):
        """
        Toggle auto-play functionality with configurable speed.
        
        Shows running frequency in Hz and start frame when auto-playing.
        """
        if self.is_playing:
            # Stop auto-play
            self.auto_play_timer.stop()
            self.is_playing = False
            self.play_pause_btn.setText("▶ Play")
            
            # Update status displays
            speed_text = self.speed_combo.currentText()
            self.playback_status_label.setText(f"Ready - {speed_text}")
            self.statusBar().showMessage("Auto-play stopped")
        else:
            # Start auto-play if we have frames
            if self.frame_files:
                self.auto_play_timer.start(self.playback_speed_ms)
                self.is_playing = True
                self.play_pause_btn.setText("⏸ Pause")
                
                # Calculate frequency and show start frame
                frequency_hz = 1000 / self.playback_speed_ms
                start_frame = self.current_frame_index + 1  # 1-based for display
                speed_text = self.speed_combo.currentText()
                
                # Update displays with frequency and start frame
                self.playback_status_label.setText(f"{frequency_hz:.1f}Hz | Start: {start_frame}")
                self.statusBar().showMessage(f"Auto-play started: {speed_text} ({frequency_hz:.1f}Hz) from frame {start_frame}")
    
    def _auto_play_step(self):
        """Advance to next frame during auto-play."""
        if self.current_frame_index < len(self.frame_files) - 1:
            self.next_frame()
        else:
            # Reached end, stop auto-play
            self.toggle_auto_play()
            self.statusBar().showMessage("Auto-play completed")
    
    def _on_speed_combo_changed(self, speed_text: str):
        """
        Handle speed combo box changes with real-time status display.
        
        Updates the auto-play timer interval based on selected speed.
        Shows running frequency in Hz and current frame when auto-playing.
        
        Args:
            speed_text: Selected speed text (e.g., "1x", "2x")
        """
        # Get the speed value in milliseconds from combo box
        speed_ms = self.speed_combo.currentData()
        if speed_ms:
            self.playback_speed_ms = speed_ms
            
            # Calculate frequency in Hz
            frequency_hz = 1000 / speed_ms
            
            # Update status display
            if self.is_playing:
                start_frame = 1  # 1-based for display
                self.playback_status_label.setText(f"{frequency_hz:.1f}Hz | Start: {start_frame}")
                self.statusBar().showMessage(f"Auto-play: {speed_text} ({frequency_hz:.1f} Hz)")
                
                # Update timer if currently playing
                self.auto_play_timer.setInterval(speed_ms)
            else:
                self.playback_status_label.setText(f"Ready - {speed_text}")
                self.statusBar().showMessage(f"Speed set to {speed_text} ({frequency_hz:.1f} Hz)")
    
    def _on_multi_frame_toggled(self, enabled: bool):
        """Handle multi-frame mode toggle."""
        self.multi_frame_mode = enabled
        self.set_start_btn.setEnabled(enabled)
        self.set_end_btn.setEnabled(enabled)
        
        # Update button labels based on multi-frame mode
        if enabled and hasattr(self, 'remove_btn') and hasattr(self, 'clear_btn'):
            self.remove_btn.setText("Remove From Range")
            self.clear_btn.setText("Clear Range")
        elif hasattr(self, 'remove_btn') and hasattr(self, 'clear_btn'):
            self.remove_btn.setText("Remove Selected")
            self.clear_btn.setText("Clear Frame")
        
        if not enabled:
            # Reset range
            self.frame_range_start = None
            self.frame_range_end = None
            self.range_display_label.setText("Range: Not set")
    
    def _set_range_start(self):
        """Set start frame for multi-frame labeling."""
        if self.multi_frame_mode:
            self.frame_range_start = self.current_frame_index
            self._update_range_display()
    
    def _set_range_end(self):
        """Set end frame for multi-frame labeling."""
        if self.multi_frame_mode:
            if self.frame_range_start is not None and self.current_frame_index < self.frame_range_start:
                QMessageBox.warning(self, "Invalid Range", "End frame must be after start frame.")
                return
            
            self.frame_range_end = self.current_frame_index
            self._update_range_display()
    
    def _update_range_display(self):
        """Update range display label."""
        if self.frame_range_start is not None and self.frame_range_end is not None:
            count = self.frame_range_end - self.frame_range_start + 1
            self.range_display_label.setText(
                f"Range: {self.frame_range_start + 1}-{self.frame_range_end + 1} ({count} frames)"
            )
        elif self.frame_range_start is not None:
            self.range_display_label.setText(f"Start: {self.frame_range_start + 1}, End: Not set")
        else:
            self.range_display_label.setText("Range: Not set")
    
    def _validate_multi_frame_range(self) -> bool:
        """
        Validate multi-frame range for correctness.
        
        This prevents crashes when start frame is behind end frame or range is invalid.
        Shows warnings for incorrect usage instead of letting the GUI crash.
        
        Returns:
            bool: True if range is valid, False otherwise
        """
        if not self.multi_frame_mode:
            return True
        
        # Check if both start and end are set
        if self.frame_range_start is None or self.frame_range_end is None:
            QMessageBox.warning(
                self, 
                "Incomplete Range", 
                "Multi-frame mode is enabled but range is not complete.\n"
                "Please set both start and end frames before proceeding."
            )
            return False
        
        # Check if start frame is behind end frame (incorrect usage)
        if self.frame_range_start > self.frame_range_end:
            QMessageBox.warning(
                self,
                "Invalid Frame Range", 
                f"Start frame ({self.frame_range_start + 1}) cannot be after end frame ({self.frame_range_end + 1}).\n"
                "Please set the start frame before the end frame."
            )
            return False
        
        # Check if range is within valid bounds
        max_frame = len(self.frame_files) - 1 if self.frame_files else 0
        if self.frame_range_start < 0 or self.frame_range_end > max_frame:
            QMessageBox.warning(
                self,
                "Range Out of Bounds",
                f"Frame range ({self.frame_range_start + 1}-{self.frame_range_end + 1}) is outside valid bounds (1-{max_frame + 1})."
            )
            return False
        
        # Additional validation: warn for very large ranges
        range_size = self.frame_range_end - self.frame_range_start + 1
        if range_size > 1000:  # Configurable threshold
            reply = QMessageBox.question(
                self,
                "Large Range Warning",
                f"You are about to apply annotation to {range_size} frames.\n"
                "This is a large range and may take some time.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            return reply == QMessageBox.Yes
        
        return True
    
    def _add_event_annotation(self):
        """Add event annotation to current frame(s)."""
        event_name = self.event_input.text().strip()
        if not event_name:
            QMessageBox.warning(self, "Invalid Input", "Please enter an event name.")
            return
        
        annotation = {
            "event": event_name
        }
        
        self._add_annotation("event", annotation)
        self.event_input.clear()  # Clear input

    def _remove_selected_event(self):
        """
        Remove selected event from current frame's event list.

        IMPROVEMENT: Allows selective removal of individual events from multi-event frames.
        """
        current_item = self.current_frame_events_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select an event to remove.")
            return

        event_name = current_item.text()

        # Find and remove the matching event annotation
        if self.current_frame_index in self.annotations:
            frame_annotations = self.annotations[self.current_frame_index]
            # Find all event annotations
            event_annotations = [a for a in frame_annotations if a["category"] == "event"]

            # Remove the matching event
            for annotation in event_annotations:
                if annotation["data"].get("event") == event_name:
                    frame_annotations.remove(annotation)
                    break

            # Clean up if no annotations left for this frame
            if not frame_annotations:
                del self.annotations[self.current_frame_index]

            # Update displays
            self._update_annotation_list()
            self.statusBar().showMessage(f"Removed event '{event_name}' from frame {self.current_frame_index + 1}")

    def _move_event_up(self):
        """
        Move selected event up in the sequence.

        IMPROVEMENT: Allows reordering of multiple events on the same frame.
        FIXED: Now properly swaps events in the original frame_annotations list.
        """
        current_row = self.current_frame_events_list.currentRow()
        if current_row <= 0:
            return  # Already at top or no selection

        if self.current_frame_index not in self.annotations:
            return

        # Get all event annotations for this frame with their indices
        frame_annotations = self.annotations[self.current_frame_index]
        event_indices = [i for i, a in enumerate(frame_annotations) if a["category"] == "event"]

        if current_row < len(event_indices):
            # FIXED: Swap in the original frame_annotations list (not a filtered copy)
            idx_current = event_indices[current_row]
            idx_previous = event_indices[current_row - 1]

            # Swap the annotations in the original list
            frame_annotations[idx_current], frame_annotations[idx_previous] = \
                frame_annotations[idx_previous], frame_annotations[idx_current]

            # Update the display and maintain selection
            self._update_annotation_list()
            self.current_frame_events_list.setCurrentRow(current_row - 1)
            self.statusBar().showMessage(f"Moved event up")

    def _move_event_down(self):
        """
        Move selected event down in the sequence.

        IMPROVEMENT: Allows reordering of multiple events on the same frame.
        FIXED: Now properly swaps events in the original frame_annotations list.
        """
        current_row = self.current_frame_events_list.currentRow()
        if current_row < 0:
            return  # No selection

        if self.current_frame_index not in self.annotations:
            return

        # Get all event annotations for this frame with their indices
        frame_annotations = self.annotations[self.current_frame_index]
        event_indices = [i for i, a in enumerate(frame_annotations) if a["category"] == "event"]

        if current_row >= len(event_indices) - 1:
            return  # Already at bottom

        # FIXED: Swap in the original frame_annotations list (not a filtered copy)
        idx_current = event_indices[current_row]
        idx_next = event_indices[current_row + 1]

        # Swap the annotations in the original list
        frame_annotations[idx_current], frame_annotations[idx_next] = \
            frame_annotations[idx_next], frame_annotations[idx_current]

        # Update the display and maintain selection
        self._update_annotation_list()
        self.current_frame_events_list.setCurrentRow(current_row + 1)
        self.statusBar().showMessage(f"Moved event down")

    def _add_phase_annotation(self):
        """Add independent phase annotations for each PSM with phase text."""
        phases_to_add = {}
        
        # Collect phase texts for each PSM
        for psm, input_widget in self.phase_inputs.items():
            phase_text = input_widget.text().strip()
            if phase_text:  # Only add if there's actual text
                phases_to_add[psm] = phase_text
        
        if not phases_to_add:
            QMessageBox.warning(self, "Invalid Input", "Please enter at least one phase.")
            return
        
        # Remove existing phase labels first, then add new ones
        self._remove_phase_labels()
        
        # Add individual phase annotation for each PSM with text
        for psm, phase_text in phases_to_add.items():
            phase_data = {"phase": {psm: phase_text}}
            self._add_annotation(f"phase_{psm}", phase_data)
        
        # Clear text inputs
        for input_widget in self.phase_inputs.values():
            input_widget.clear()
    
    def _remove_phase_labels(self):
        """Remove all existing phase labels for the current frame(s)."""
        # Validate multi-frame range before proceeding to prevent crashes
        if not self._validate_multi_frame_range():
            return  # Validation failed, show warning and abort operation
        
        if self.multi_frame_mode and self.frame_range_start is not None and self.frame_range_end is not None:
            frames_to_process = list(range(self.frame_range_start, self.frame_range_end + 1))
        else:
            frames_to_process = [self.current_frame_index]
        
        for frame_idx in frames_to_process:
            if frame_idx in self.annotations:
                # Remove all phase-related annotations (phase and phase_PSMx)
                self.annotations[frame_idx] = [
                    annotation for annotation in self.annotations[frame_idx] 
                    if not (annotation["category"] == "phase" or 
                           annotation["category"].startswith("phase_"))
                ]
                # If no annotations left for this frame, remove the frame entry
                if not self.annotations[frame_idx]:
                    del self.annotations[frame_idx]
        
        # Update display
        self._update_annotation_list()
        self._update_statistics()
    
    def _add_contact_annotation(self):
        """
        Add independent contact annotations for each PSM.

        FIXED: When no PSMs selected, adds non-contact states (all PSMs = 0) instead of removing.

        Creates individual contact annotations that can be added/removed independently:
        - Each PSM contact is a separate annotation
        - Can add/remove individual PSM contacts without affecting others
        - Still saves in combined JSON format for compatibility

        Behavior:
        - No PSMs selected → Add non-contact states for ALL PSMs (contact = 0)
        - Some PSMs selected → Add contact states (contact = 1) for selected, non-contact (0) for others

        Display format: Shows individual lines for each PSM:
        ├── [contact] PSM1: ✓ Contact  (if contact = 1)
        ├── [contact] PSM2: ○ No Contact  (if contact = 0)
        └── [contact] PSM3: ✓ Contact  (if contact = 1)
        """
        # CRITICAL FIX: Validate that active_psms is not empty
        # This prevents the bug where removing contacts without adding them back
        if not self.active_psms:
            QMessageBox.warning(
                self,
                "No PSMs Configured",
                "No PSMs are configured. Please load a configuration file first.\n\n"
                "File → Load Configuration"
            )
            print("ERROR: Cannot add contact annotation - no active PSMs configured")
            return

        # CRITICAL FIX: Validate that contact_checkboxes is properly initialized
        if not self.contact_checkboxes:
            QMessageBox.warning(
                self,
                "Contact Section Not Ready",
                "Contact detection section is not properly initialized.\n\n"
                "Please load a configuration file first: File → Load Configuration"
            )
            print("ERROR: Cannot add contact annotation - contact_checkboxes not initialized")
            return

        # Get currently selected PSMs
        selected_psms = []
        for psm, checkbox in self.contact_checkboxes.items():
            if checkbox.isChecked():
                selected_psms.append(psm)

        # FIXED: Always remove existing contact labels first to ensure clean state
        self._remove_contact_labels()

        # FIXED: Add annotations for ALL active PSMs (not just selected ones)
        # This ensures both contact and non-contact states are properly tracked
        for psm in self.active_psms:
            if psm in selected_psms:
                # PSM is selected → Add contact (value = 1)
                contact_data = {psm: 1}
            else:
                # PSM is NOT selected → Add non-contact (value = 0)
                # FIXED: Previously, unselected PSMs were not added at all
                contact_data = {psm: 0}

            self._add_annotation(f"contact_{psm}", contact_data)

        # Clear checkboxes
        for checkbox in self.contact_checkboxes.values():
            checkbox.setChecked(False)
    
    
    def _remove_contact_labels(self):
        """Remove all existing contact labels for the current frame(s)."""
        # Validate multi-frame range before proceeding to prevent crashes
        if not self._validate_multi_frame_range():
            return  # Validation failed, show warning and abort operation
        
        if self.multi_frame_mode and self.frame_range_start is not None and self.frame_range_end is not None:
            frames_to_process = list(range(self.frame_range_start, self.frame_range_end + 1))
        else:
            frames_to_process = [self.current_frame_index]
        
        for frame_idx in frames_to_process:
            if frame_idx in self.annotations:
                # Remove all contact-related annotations (contact_detection and contact_PSMx)
                self.annotations[frame_idx] = [
                    annotation for annotation in self.annotations[frame_idx] 
                    if not (annotation["category"] == "contact_detection" or 
                           annotation["category"].startswith("contact_"))
                ]
                # If no annotations left for this frame, remove the frame entry
                if not self.annotations[frame_idx]:
                    del self.annotations[frame_idx]
        
        # Update display
        self._update_annotation_list()
        self._update_statistics()
        
        # Status message
        if len(frames_to_process) > 1:
            self.statusBar().showMessage(f"Removed all contact labels from {len(frames_to_process)} frames")
        else:
            self.statusBar().showMessage(f"Removed all contact labels from frame {frames_to_process[0] + 1}")
    
    def _add_annotation(self, category: str, annotation_data: Union[Dict, str]):
        """
        Add annotation to frame(s) and save to appropriate folder structure.
        
        Implements the folder structure:
        - save in subfolder `annotation`
        - create sub-subfolder for category (contact_detection, event, phase)
        - save individual JSON files per frame (0-n.json)
        
        Args:
            category: Annotation category (event, phase, contact_detection)
            annotation_data: The annotation data
        """
        # Validate multi-frame range before proceeding to prevent crashes
        if not self._validate_multi_frame_range():
            return  # Validation failed, show warning and abort operation
        
        if self.multi_frame_mode and self.frame_range_start is not None and self.frame_range_end is not None:
            # Multi-frame annotation
            frames_to_annotate = list(range(self.frame_range_start, self.frame_range_end + 1))
            
            # Confirm multi-frame operation
            reply = QMessageBox.question(
                self,
                "Multi-frame Annotation",
                f"Apply {category} annotation to {len(frames_to_annotate)} frames "
                f"({self.frame_range_start + 1}-{self.frame_range_end + 1})?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        else:
            # Single frame annotation
            frames_to_annotate = [self.current_frame_index]
        
        # Add to internal storage
        for frame_idx in frames_to_annotate:
            if frame_idx not in self.annotations:
                self.annotations[frame_idx] = []
            
            self.annotations[frame_idx].append({
                "category": category,
                "data": annotation_data,
                "timestamp": frame_idx
            })
        
        # Note: No auto-save - user must explicitly click Save button
        
        # Update display
        self._update_annotation_list()
        self._update_statistics()
        
        # Status message
        if len(frames_to_annotate) > 1:
            self.statusBar().showMessage(f"Added {category} annotation to {len(frames_to_annotate)} frames")
        else:
            self.statusBar().showMessage(f"Added {category} annotation to frame {frames_to_annotate[0] + 1}")
    
    def _load_existing_annotations(self):
        """
        Load existing annotation files from the annotation folder structure.
        
        Searches for existing annotation folders (contact_detection, event, phase)
        and loads all available annotations into memory. This allows users to 
        continue working with previously saved annotations.
        
        Expected folder structure:
        <save_folder>/annotation/
        ├── contact_detection/
        │   ├── 0.json
        │   ├── 1.json
        │   └── ...
        ├── event/
        │   ├── 0.json
        │   ├── 1.json
        │   └── ...
        └── phase/
            ├── 0.json
            ├── 1.json
            └── ...
        """
        try:
            if not self.config:
                return
                
            # Get the annotation base folder
            base_folder = self.get_save_folder()
            annotation_folder = base_folder / "annotation"
            
            if not annotation_folder.exists():
                print(f"No existing annotation folder found at: {annotation_folder}")
                return
            
            print(f"Loading existing annotations from: {annotation_folder}")
            
            # Categories to load
            categories = ['contact_detection', 'event', 'phase']
            loaded_counts = {}
            
            for category in categories:
                category_folder = annotation_folder / category
                if not category_folder.exists():
                    continue
                
                # Find all JSON files in this category
                json_files = list(category_folder.glob("*.json"))
                loaded_counts[category] = 0
                
                for json_file in json_files:
                    try:
                        # Extract frame index from filename (e.g., "0.json" -> 0)
                        frame_index = int(json_file.stem)
                        
                        # Load the annotation data
                        with open(json_file, 'r') as f:
                            annotation_data = json.load(f)
                        
                        # Store in our annotations dictionary
                        if frame_index not in self.annotations:
                            self.annotations[frame_index] = []
                        
                        # Convert combined saved format back to individual PSM annotations for consistency
                        # This ensures loaded annotations display the same way as before saving
                        if category == "contact_detection":
                            # ROBUST LOADING: Handle null values, 0 values, and validate data types
                            # Convert combined contact format to individual PSM annotations
                            if not isinstance(annotation_data, dict):
                                print(f"Warning: Invalid contact data in {json_file}, expected dict, got {type(annotation_data)}")
                                continue

                            for psm, contact_value in annotation_data.items():
                                # Validate and normalize contact value
                                # Handles: int (0, 1), bool (False, True), string ("0", "1"), None, null
                                try:
                                    if contact_value is None:
                                        contact_value = 0  # null → 0 (no contact, not annotated)
                                    elif isinstance(contact_value, bool):
                                        contact_value = 1 if contact_value else 0
                                    elif isinstance(contact_value, str):
                                        contact_value = int(contact_value)
                                    else:
                                        contact_value = int(contact_value)

                                    # Clamp to valid range [0, 1] in case of invalid values like 2, -1
                                    contact_value = max(0, min(1, contact_value))
                                except (ValueError, TypeError) as e:
                                    print(f"Warning: Invalid contact value '{contact_value}' for {psm} in {json_file}, defaulting to 0")
                                    contact_value = 0

                                # IMPROVEMENT: Load BOTH contact (1) and non-contact (0) states
                                # This allows GUI to reflect both states in the label indicator
                                individual_category = f"contact_{psm}"
                                individual_data = {psm: contact_value}

                                annotation_entry = {
                                    'category': individual_category,
                                    'data': individual_data,
                                    'timestamp': datetime.datetime.now().isoformat()
                                }

                                # Check if this individual annotation already exists
                                existing_entry = None
                                for i, entry in enumerate(self.annotations[frame_index]):
                                    if entry['category'] == individual_category:
                                        existing_entry = i
                                        break

                                if existing_entry is not None:
                                    # Replace existing annotation
                                    self.annotations[frame_index][existing_entry] = annotation_entry
                                else:
                                    # Add new annotation
                                    self.annotations[frame_index].append(annotation_entry)
                                        
                        elif category == "phase":
                            # ROBUST LOADING: Handle null/None values and validate data types
                            # Convert combined phase format to individual PSM annotations
                            if not isinstance(annotation_data, dict):
                                print(f"Warning: Invalid phase data in {json_file}, expected dict, got {type(annotation_data)}")
                                continue

                            phase_data = annotation_data.get("phase", {})
                            if not isinstance(phase_data, dict):
                                print(f"Warning: Invalid phase data structure in {json_file}, expected dict, got {type(phase_data)}")
                                continue

                            for psm, phase_value in phase_data.items():
                                # Only load phase annotations for PSMs with actual phase text
                                # Handles: None, null, empty string "", whitespace-only strings
                                # Treats all as "not annotated" and skips loading them
                                if phase_value is None:
                                    continue  # null/None → not annotated, skip
                                if not isinstance(phase_value, str):
                                    phase_value = str(phase_value) if phase_value else None
                                if not phase_value or not phase_value.strip():
                                    continue  # Empty or whitespace-only → not annotated, skip

                                if phase_value:  # Only add phase annotations for PSMs with actual phase text
                                    individual_category = f"phase_{psm}"
                                    individual_data = {"phase": {psm: phase_value}}
                                    
                                    annotation_entry = {
                                        'category': individual_category,
                                        'data': individual_data,
                                        'timestamp': datetime.datetime.now().isoformat()
                                    }
                                    
                                    # Check if this individual annotation already exists
                                    existing_entry = None
                                    for i, entry in enumerate(self.annotations[frame_index]):
                                        if entry['category'] == individual_category:
                                            existing_entry = i
                                            break
                                    
                                    if existing_entry is not None:
                                        # Replace existing annotation
                                        self.annotations[frame_index][existing_entry] = annotation_entry
                                    else:
                                        # Add new annotation
                                        self.annotations[frame_index].append(annotation_entry)
                        
                        else:
                            # Handle other categories (like event) normally
                            if category == 'event':
                                # ROBUST LOADING: Handle null/None values for events
                                # Handle both old and new event JSON formats
                                if isinstance(annotation_data, dict) and "events" in annotation_data:
                                    # New format: {"events": ["test1", "test2"]} - multiple events
                                    events_list = annotation_data["events"]
                                    if not isinstance(events_list, list):
                                        print(f"Warning: Invalid events format in {json_file}, expected list, got {type(events_list)}")
                                        continue

                                    for event_label in events_list:
                                        # Skip null, None, or empty event labels
                                        if event_label is None or not str(event_label).strip():
                                            continue  # null/None/empty → not annotated, skip

                                        annotation_entry = {
                                            'category': category,
                                            'data': {'event': str(event_label).strip()},
                                            'timestamp': datetime.datetime.now().isoformat()
                                        }
                                        # Always add new event annotations - enable multiple event labels per frame
                                        self.annotations[frame_index].append(annotation_entry)

                                elif isinstance(annotation_data, dict) and "event" in annotation_data:
                                    # Old format: {"event": "test1"} - single event
                                    event_value = annotation_data["event"]

                                    # Skip null, None, or empty event values
                                    if event_value is None or not str(event_value).strip():
                                        continue  # null/None/empty → not annotated, skip

                                    annotation_entry = {
                                        'category': category,
                                        'data': {'event': str(event_value).strip()},
                                        'timestamp': datetime.datetime.now().isoformat()
                                    }
                                    # Always add new event annotations - enable multiple event labels per frame
                                    self.annotations[frame_index].append(annotation_entry)
                                else:
                                    # Fallback for any other event format
                                    annotation_entry = {
                                        'category': category,
                                        'data': {'event': str(annotation_data)},
                                        'timestamp': datetime.datetime.now().isoformat()
                                    }
                                    self.annotations[frame_index].append(annotation_entry)
                            else:
                                # For non-event categories, use original logic
                                annotation_entry = {
                                    'category': category,
                                    'data': annotation_data,
                                    'timestamp': datetime.datetime.now().isoformat()
                                }
                                
                                # For non-event categories, check if annotation already exists
                                existing_entry = None
                                for i, entry in enumerate(self.annotations[frame_index]):
                                    if entry['category'] == category:
                                        existing_entry = i
                                        break
                                
                                if existing_entry is not None:
                                    # Replace existing annotation for non-event categories
                                    self.annotations[frame_index][existing_entry] = annotation_entry
                                else:
                                    # Add new annotation
                                    self.annotations[frame_index].append(annotation_entry)
                        
                        loaded_counts[category] += 1
                        
                    except (ValueError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                        print(f"Error loading annotation file {json_file}: {e}")
                        print(f"Traceback: {traceback.format_exc()}")
                        continue
            
            # Update status with loaded annotation counts
            total_loaded = sum(loaded_counts.values())
            if total_loaded > 0:
                status_parts = []
                for category, count in loaded_counts.items():
                    if count > 0:
                        status_parts.append(f"{category}: {count}")
                
                status_msg = f"Loaded existing annotations: {', '.join(status_parts)} (Total: {total_loaded})"
                self.statusBar().showMessage(status_msg)
                print(status_msg)
                
                # Update the annotation display for current frame
                self._update_annotation_list()
            else:
                print("No existing annotations found to load")
                
        except Exception as e:
            print(f"Error loading existing annotations: {e}")
            # Don't show error to user - this is optional functionality
    
    def load_existing_annotations_dialog(self):
        """
        Open a dialog to allow user to select the annotation folder location.
        
        This method provides a file dialog for users to browse and select
        the location of existing annotation folders, giving them flexibility
        to load annotations from any location instead of just the config save folder.
        """
        try:
            # Open folder selection dialog
            annotation_folder = QFileDialog.getExistingDirectory(
                self,
                "Select Annotation Folder",
                str(Path.home()),  # Start from home directory
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            
            if not annotation_folder:
                return
            
            annotation_path = Path(annotation_folder)
            
            # Check if this looks like a valid annotation folder
            # It should contain at least one of: contact_detection, event, phase subfolders
            expected_subfolders = ['contact_detection', 'event', 'phase']
            found_subfolders = []
            
            for subfolder in expected_subfolders:
                if (annotation_path / subfolder).exists():
                    found_subfolders.append(subfolder)
            
            if not found_subfolders:
                # Ask user if they want to continue anyway
                response = QMessageBox.question(
                    self,
                    "No Annotation Subfolders Found",
                    f"The selected folder doesn't contain the expected annotation subfolders:\n"
                    f"{', '.join(expected_subfolders)}\n\n"
                    f"Do you want to continue loading from this location anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if response == QMessageBox.No:
                    return
            else:
                # Inform user what was found
                subfolder_text = ', '.join(found_subfolders)
                QMessageBox.information(
                    self,
                    "Annotation Subfolders Found",
                    f"Found annotation subfolders: {subfolder_text}\n\n"
                    f"Loading annotations from: {annotation_path}"
                )
            
            # Load annotations from the selected folder
            self._load_existing_annotations_from_folder(annotation_path)
            
        except Exception as e:
            error_msg = f"Error selecting annotation folder: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "Annotation Loading Error", error_msg)
    
    def _load_existing_annotations_from_folder(self, annotation_folder: Path):
        """
        Load existing annotation files from a specific folder location.
        
        This is a modified version of _load_existing_annotations that loads
        from a user-specified folder instead of the config save folder.
        
        Args:
            annotation_folder: Path to the annotation folder containing subfolders
        """
        try:
            if not annotation_folder.exists():
                QMessageBox.warning(self, "Folder Not Found", f"Annotation folder not found: {annotation_folder}")
                return
            
            print(f"Loading existing annotations from: {annotation_folder}")
            
            # Categories to load
            categories = ['contact_detection', 'event', 'phase']
            loaded_counts = {}
            
            for category in categories:
                category_folder = annotation_folder / category
                if not category_folder.exists():
                    continue
                
                # Find all JSON files in this category
                json_files = list(category_folder.glob("*.json"))
                loaded_counts[category] = 0
                
                for json_file in json_files:
                    try:
                        # Extract frame index from filename (e.g., "0.json" -> 0)
                        frame_index = int(json_file.stem)
                        
                        # Load the annotation data
                        with open(json_file, 'r') as f:
                            annotation_data = json.load(f)
                        
                        # Store in our annotations dictionary
                        if frame_index not in self.annotations:
                            self.annotations[frame_index] = []
                        
                        # Convert combined saved format back to individual PSM annotations for consistency
                        # This ensures loaded annotations display the same way as before saving
                        if category == "contact_detection":
                            # ROBUST LOADING: Handle null values, 0 values, and validate data types
                            # Convert combined contact format to individual PSM annotations
                            if not isinstance(annotation_data, dict):
                                print(f"Warning: Invalid contact data in {json_file}, expected dict, got {type(annotation_data)}")
                                continue

                            for psm, contact_value in annotation_data.items():
                                # Validate and normalize contact value
                                # Handles: int (0, 1), bool (False, True), string ("0", "1"), None, null
                                try:
                                    if contact_value is None:
                                        contact_value = 0  # null → 0 (no contact, not annotated)
                                    elif isinstance(contact_value, bool):
                                        contact_value = 1 if contact_value else 0
                                    elif isinstance(contact_value, str):
                                        contact_value = int(contact_value)
                                    else:
                                        contact_value = int(contact_value)

                                    # Clamp to valid range [0, 1] in case of invalid values like 2, -1
                                    contact_value = max(0, min(1, contact_value))
                                except (ValueError, TypeError) as e:
                                    print(f"Warning: Invalid contact value '{contact_value}' for {psm} in {json_file}, defaulting to 0")
                                    contact_value = 0

                                # IMPROVEMENT: Load BOTH contact (1) and non-contact (0) states
                                # This allows GUI to reflect both states in the label indicator
                                individual_category = f"contact_{psm}"
                                individual_data = {psm: contact_value}

                                annotation_entry = {
                                    'category': individual_category,
                                    'data': individual_data,
                                    'timestamp': datetime.datetime.now().isoformat()
                                }

                                # Check if this individual annotation already exists
                                existing_entry = None
                                for i, entry in enumerate(self.annotations[frame_index]):
                                    if entry['category'] == individual_category:
                                        existing_entry = i
                                        break

                                if existing_entry is not None:
                                    # Replace existing annotation
                                    self.annotations[frame_index][existing_entry] = annotation_entry
                                else:
                                    # Add new annotation
                                    self.annotations[frame_index].append(annotation_entry)
                                        
                        elif category == "phase":
                            # ROBUST LOADING: Handle null/None values and validate data types
                            # Convert combined phase format to individual PSM annotations
                            if not isinstance(annotation_data, dict):
                                print(f"Warning: Invalid phase data in {json_file}, expected dict, got {type(annotation_data)}")
                                continue

                            phase_data = annotation_data.get("phase", {})
                            if not isinstance(phase_data, dict):
                                print(f"Warning: Invalid phase data structure in {json_file}, expected dict, got {type(phase_data)}")
                                continue

                            for psm, phase_value in phase_data.items():
                                # Only load phase annotations for PSMs with actual phase text
                                # Handles: None, null, empty string "", whitespace-only strings
                                # Treats all as "not annotated" and skips loading them
                                if phase_value is None:
                                    continue  # null/None → not annotated, skip
                                if not isinstance(phase_value, str):
                                    phase_value = str(phase_value) if phase_value else None
                                if not phase_value or not phase_value.strip():
                                    continue  # Empty or whitespace-only → not annotated, skip

                                if phase_value:  # Only add phase annotations for PSMs with actual phase text
                                    individual_category = f"phase_{psm}"
                                    individual_data = {"phase": {psm: phase_value}}
                                    
                                    annotation_entry = {
                                        'category': individual_category,
                                        'data': individual_data,
                                        'timestamp': datetime.datetime.now().isoformat()
                                    }
                                    
                                    # Check if this individual annotation already exists
                                    existing_entry = None
                                    for i, entry in enumerate(self.annotations[frame_index]):
                                        if entry['category'] == individual_category:
                                            existing_entry = i
                                            break
                                    
                                    if existing_entry is not None:
                                        # Replace existing annotation
                                        self.annotations[frame_index][existing_entry] = annotation_entry
                                    else:
                                        # Add new annotation
                                        self.annotations[frame_index].append(annotation_entry)
                        
                        else:
                            # Handle other categories (like event) normally
                            if category == 'event':
                                # ROBUST LOADING: Handle null/None values for events
                                # Handle both old and new event JSON formats
                                if isinstance(annotation_data, dict) and "events" in annotation_data:
                                    # New format: {"events": ["test1", "test2"]} - multiple events
                                    events_list = annotation_data["events"]
                                    if not isinstance(events_list, list):
                                        print(f"Warning: Invalid events format in {json_file}, expected list, got {type(events_list)}")
                                        continue

                                    for event_label in events_list:
                                        # Skip null, None, or empty event labels
                                        if event_label is None or not str(event_label).strip():
                                            continue  # null/None/empty → not annotated, skip

                                        annotation_entry = {
                                            'category': category,
                                            'data': {'event': str(event_label).strip()},
                                            'timestamp': datetime.datetime.now().isoformat()
                                        }
                                        # Always add new event annotations - enable multiple event labels per frame
                                        self.annotations[frame_index].append(annotation_entry)

                                elif isinstance(annotation_data, dict) and "event" in annotation_data:
                                    # Old format: {"event": "test1"} - single event
                                    event_value = annotation_data["event"]

                                    # Skip null, None, or empty event values
                                    if event_value is None or not str(event_value).strip():
                                        continue  # null/None/empty → not annotated, skip

                                    annotation_entry = {
                                        'category': category,
                                        'data': {'event': str(event_value).strip()},
                                        'timestamp': datetime.datetime.now().isoformat()
                                    }
                                    # Always add new event annotations - enable multiple event labels per frame
                                    self.annotations[frame_index].append(annotation_entry)
                                else:
                                    # Fallback for any other event format
                                    annotation_entry = {
                                        'category': category,
                                        'data': {'event': str(annotation_data)},
                                        'timestamp': datetime.datetime.now().isoformat()
                                    }
                                    self.annotations[frame_index].append(annotation_entry)
                            else:
                                # For non-event categories, use original logic
                                annotation_entry = {
                                    'category': category,
                                    'data': annotation_data,
                                    'timestamp': datetime.datetime.now().isoformat()
                                }
                                
                                # For non-event categories, check if annotation already exists
                                existing_entry = None
                                for i, entry in enumerate(self.annotations[frame_index]):
                                    if entry['category'] == category:
                                        existing_entry = i
                                        break
                                
                                if existing_entry is not None:
                                    # Replace existing annotation for non-event categories
                                    self.annotations[frame_index][existing_entry] = annotation_entry
                                else:
                                    # Add new annotation
                                    self.annotations[frame_index].append(annotation_entry)
                        
                        loaded_counts[category] += 1
                        
                    except (ValueError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                        print(f"Error loading annotation file {json_file}: {e}")
                        print(f"Traceback: {traceback.format_exc()}")
                        continue
            
            # Update status with loaded annotation counts
            total_loaded = sum(loaded_counts.values())
            if total_loaded > 0:
                status_parts = []
                for category, count in loaded_counts.items():
                    if count > 0:
                        status_parts.append(f"{category}: {count}")
                
                status_msg = f"Loaded annotations from {annotation_folder.name}: {', '.join(status_parts)} (Total: {total_loaded})"
                self.statusBar().showMessage(status_msg)
                print(status_msg)
                
                # Show success message to user
                QMessageBox.information(
                    self,
                    "Annotations Loaded Successfully",
                    f"Successfully loaded {total_loaded} annotations from:\n{annotation_folder}\n\n"
                    f"Breakdown: {', '.join(status_parts)}"
                )
                
                # Update the annotation display for current frame
                self._update_annotation_list()
            else:
                print("No existing annotations found to load")
                QMessageBox.information(
                    self,
                    "No Annotations Found",
                    f"No valid annotation files were found in:\n{annotation_folder}\n\n"
                    f"Expected subfolders: contact_detection, event, phase"
                )
                
        except Exception as e:
            error_msg = f"Error loading existing annotations: {e}"
            print(error_msg)
            QMessageBox.critical(self, "Annotation Loading Error", error_msg)
    
    def _update_annotation_list(self):
        """
        Update the annotation list for current frame.

        IMPROVEMENT: Also updates the current frame events list for multi-event management.
        """
        self.annotation_list.clear()

        # IMPROVEMENT: Update current frame events list for multi-event UI
        if hasattr(self, 'current_frame_events_list'):
            self.current_frame_events_list.clear()

        if self.current_frame_index in self.annotations:
            for annotation in self.annotations[self.current_frame_index]:
                category = annotation["category"]
                data = annotation["data"]
                
                # Create display text for individual annotations (since we now convert during loading)
                if category == "event":
                    event_name = data['event']
                    display_text = f"[{category}] {event_name}"

                    # IMPROVEMENT: Also add to current frame events list for multi-event management
                    if hasattr(self, 'current_frame_events_list'):
                        self.current_frame_events_list.addItem(event_name)
                elif category.startswith("phase_"):
                    # Handle individual PSM phase annotations (phase_PSM1, phase_PSM2, etc.)
                    psm = category.replace("phase_", "")
                    if "phase" in data and psm in data["phase"] and data["phase"][psm]:
                        display_text = f"[phase] {psm}: {data['phase'][psm]}"
                    else:
                        continue  # Skip if no phase text
                elif category.startswith("contact_"):
                    # IMPROVEMENT: Show BOTH contact and non-contact states
                    # Handle individual PSM contact annotations (contact_PSM1, contact_PSM2, etc.)
                    psm = category.replace("contact_", "")
                    contact_value = data.get(psm, 0)

                    if contact_value == 1:
                        display_text = f"[contact] {psm}: ✓ Contact"
                    else:
                        display_text = f"[contact] {psm}: ○ No Contact"
                else:
                    display_text = f"[{category}] {str(data)}"
                
                # Create list item with enhanced styling like annotate_event.py
                list_item = QListWidgetItem(display_text)
                
                # Add category-based color coding
                if category == "event":
                    list_item.setBackground(QColor(220, 255, 220))  # Light green
                elif category.startswith("phase_"):
                    list_item.setBackground(QColor(220, 220, 255))  # Light blue
                elif category.startswith("contact_"):
                    # Different colors for contact vs non-contact
                    psm = category.replace("contact_", "")
                    contact_value = data.get(psm, 0)
                    if contact_value == 1:
                        list_item.setBackground(QColor(255, 200, 200))  # Light red (contact)
                    else:
                        list_item.setBackground(QColor(240, 240, 240))  # Light gray (no contact)
                
                self.annotation_list.addItem(list_item)

        # IMPROVEMENT: Update visual indicators when annotation list changes
        self._update_visual_indicators()

    def _update_visual_indicators(self):
        """
        Update all visual annotation indicators including progress bar and current frame status.

        This method provides real-time visual feedback about annotation progress:
        - Progress bar showing % of frames annotated
        - Current frame status (annotated or not)
        - Annotation progress label with count
        """
        if not self.frame_files:
            return

        total_frames = len(self.frame_files)
        annotated_frames = len(self.annotations)
        progress_percent = (annotated_frames / total_frames * 100) if total_frames > 0 else 0

        # Update progress bar
        if hasattr(self, 'annotation_progress_bar'):
            self.annotation_progress_bar.setValue(int(progress_percent))

        # Update progress label
        if hasattr(self, 'annotation_progress_label'):
            self.annotation_progress_label.setText(
                f"{annotated_frames}/{total_frames} frames"
            )

        # Update current frame status indicator
        if hasattr(self, 'current_frame_status_label'):
            if self.current_frame_index in self.annotations:
                # Frame has annotations - show green check mark
                self.current_frame_status_label.setText("✓")
                self.current_frame_status_label.setStyleSheet(
                    "font-size: 16px; font-weight: bold; color: #4CAF50; padding: 3px;"
                )
                self.current_frame_status_label.setToolTip(
                    f"Frame {self.current_frame_index + 1} has {len(self.annotations[self.current_frame_index])} annotation(s)"
                )
            else:
                # Frame has no annotations - show empty circle
                self.current_frame_status_label.setText("○")
                self.current_frame_status_label.setStyleSheet(
                    "font-size: 16px; font-weight: bold; color: #999; padding: 3px;"
                )
                self.current_frame_status_label.setToolTip(
                    f"Frame {self.current_frame_index + 1} has no annotations"
                )

    def _update_statistics(self):
        """Update statistics display, similar to annotate_event.py."""
        if not self.annotations:
            self.stats_label.setText("No annotations yet")
            return
        
        # Calculate basic statistics
        total_annotations = sum(len(annotations) for annotations in self.annotations.values())
        annotated_frames = len(self.annotations)
        total_frames = len(self.frame_files) if self.frame_files else 0
        
        # Count by category
        category_counts = {}
        for frame_annotations in self.annotations.values():
            for annotation in frame_annotations:
                category = annotation["category"]
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Create statistics display text similar to annotate_event.py
        progress_percent = (annotated_frames / max(total_frames, 1)) * 100
        
        stats_text = f"""Total Annotations: {total_annotations}
Annotated Frames: {annotated_frames}/{total_frames}
Progress: {progress_percent:.1f}%

By Category:"""
        
        for category, count in sorted(category_counts.items()):
            stats_text += f"\n  {category}: {count}"
        
        # Add PSM-specific contact statistics
        psm_contact_counts = {}
        for frame_annotations in self.annotations.values():
            for annotation in frame_annotations:
                if annotation["category"] == "contact_detection":
                    data = annotation["data"]
                    # New format: {"PSM1": 1, "PSM2": 0} - count PSMs with contact (1)
                    for psm, has_contact in data.items():
                        if has_contact == 1:  # Only count actual contact labels
                            psm_contact_counts[psm] = psm_contact_counts.get(psm, 0) + 1
        
        if psm_contact_counts:
            stats_text += "\n\nPSM Contact Labels:"
            for psm, count in sorted(psm_contact_counts.items()):
                stats_text += f"\n  {psm}: {count} contact labels"
        
        self.stats_label.setText(stats_text)
    
    def _reset_statistics(self):
        """Reset/reload statistics display."""
        self._update_statistics()
        self.statusBar().showMessage("Statistics refreshed")
    
    def _edit_annotation(self, item: QListWidgetItem):
        """
        Handle annotation editing (double-click).

        IMPROVEMENT: Allows in-place editing of annotations for quick modifications.
        """
        if not item:
            return

        # Get the annotation index from the list
        row = self.annotation_list.row(item)

        if self.current_frame_index not in self.annotations:
            return

        frame_annotations = self.annotations[self.current_frame_index]
        if row >= len(frame_annotations):
            return

        annotation = frame_annotations[row]
        category = annotation["category"]
        data = annotation["data"]

        # Create edit dialog based on annotation type
        if category == "event":
            # Edit event name
            current_event = data.get("event", "")
            new_event, ok = self._show_input_dialog(
                "Edit Event",
                "Event name:",
                current_event
            )
            if ok and new_event.strip():
                data["event"] = new_event.strip()
                self._update_annotation_list()
                self.statusBar().showMessage(f"Updated event to '{new_event.strip()}'")

        elif category.startswith("phase_"):
            # Edit phase text for specific PSM
            psm = category.replace("phase_", "")
            current_phase = data.get("phase", {}).get(psm, "")
            new_phase, ok = self._show_input_dialog(
                f"Edit Phase for {psm}",
                f"{psm} phase:",
                current_phase
            )
            if ok:
                if "phase" not in data:
                    data["phase"] = {}
                data["phase"][psm] = new_phase.strip() if new_phase.strip() else None
                self._update_annotation_list()
                self.statusBar().showMessage(f"Updated {psm} phase to '{new_phase.strip()}'")

        elif category.startswith("contact_"):
            # Contact is boolean - toggle it
            psm = category.replace("contact_", "")
            current_contact = data.get(psm, 0)
            new_contact = 0 if current_contact == 1 else 1
            data[psm] = new_contact
            self._update_annotation_list()
            status_text = "contact" if new_contact == 1 else "no contact"
            self.statusBar().showMessage(f"Updated {psm} to '{status_text}'")

        else:
            QMessageBox.information(
                self,
                "Edit Not Supported",
                f"Editing for category '{category}' is not yet supported."
            )

    def _show_input_dialog(self, title: str, label: str, default_value: str = "") -> Tuple[str, bool]:
        """
        Show an input dialog for text input.

        Args:
            title: Dialog title
            label: Input label
            default_value: Default value for the input

        Returns:
            Tuple of (input_text, ok_pressed)
        """
        from PyQt5.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(
            self,
            title,
            label,
            QLineEdit.Normal,
            default_value
        )
        return text, ok
    
    def _remove_selected_annotation(self):
        """Remove selected annotation from current frame or frame range if multi-frame mode is enabled."""
        current_item = self.annotation_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select an annotation to remove.")
            return
        
        # Validate multi-frame range before proceeding to prevent crashes
        if not self._validate_multi_frame_range():
            return  # Validation failed, show warning and abort operation
        
        if self.multi_frame_mode and self.frame_range_start is not None and self.frame_range_end is not None:
            # Multi-frame removal mode
            item_index = self.annotation_list.currentRow()
            if (self.current_frame_index in self.annotations and 
                0 <= item_index < len(self.annotations[self.current_frame_index])):
                
                # Get the annotation to match across frames
                selected_annotation = self.annotations[self.current_frame_index][item_index]
                annotation_text = selected_annotation["data"]
                category = selected_annotation["category"]
                
                # Confirm multi-frame removal
                num_frames = self.frame_range_end - self.frame_range_start + 1
                reply = QMessageBox.question(
                    self,
                    "Multi-frame Removal",
                    f"Remove matching annotation from {num_frames} frames ({self.frame_range_start + 1}-{self.frame_range_end + 1})?\n\n"
                    f"Category: {category}\nAnnotation: {str(annotation_text)[:100]}{'...' if len(str(annotation_text)) > 100 else ''}",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    removed_count = 0
                    for frame_idx in range(self.frame_range_start, self.frame_range_end + 1):
                        if frame_idx in self.annotations:
                            # Find and remove matching annotations
                            annotations_to_remove = []
                            for i, annotation in enumerate(self.annotations[frame_idx]):
                                if (annotation["category"] == category and 
                                    annotation["data"] == annotation_text):
                                    annotations_to_remove.append(i)
                            
                            # Remove in reverse order to maintain indices
                            for i in reversed(annotations_to_remove):
                                del self.annotations[frame_idx][i]
                                removed_count += 1
                            
                            # Clean up empty frame entries
                            if not self.annotations[frame_idx]:
                                del self.annotations[frame_idx]
                    
                    # Update display
                    self._update_annotation_list()
                    self._update_statistics()
                    self.statusBar().showMessage(f"Removed {removed_count} matching annotations from {num_frames} frames")
        else:
            # Single frame removal (original behavior)
            item_index = self.annotation_list.currentRow()
            if (self.current_frame_index in self.annotations and 
                0 <= item_index < len(self.annotations[self.current_frame_index])):
                
                # Remove from internal storage
                del self.annotations[self.current_frame_index][item_index]
                
                # Clean up empty frame entries
                if not self.annotations[self.current_frame_index]:
                    del self.annotations[self.current_frame_index]
                
                # Update display
                self._update_annotation_list()
                self._update_statistics()
                self.statusBar().showMessage("Annotation removed")
    
    def _clear_frame_annotations(self):
        """Clear all annotations for current frame or frame range if multi-frame mode is enabled."""
        # Validate multi-frame range before proceeding to prevent crashes
        if not self._validate_multi_frame_range():
            return  # Validation failed, show warning and abort operation
        
        if self.multi_frame_mode and self.frame_range_start is not None and self.frame_range_end is not None:
            # Multi-frame clearing mode
            num_frames = self.frame_range_end - self.frame_range_start + 1
            
            # Count total annotations in range
            total_annotations = 0
            for frame_idx in range(self.frame_range_start, self.frame_range_end + 1):
                if frame_idx in self.annotations:
                    total_annotations += len(self.annotations[frame_idx])
            
            if total_annotations == 0:
                QMessageBox.information(self, "No Annotations", "No annotations found in the selected frame range.")
                return
            
            reply = QMessageBox.question(
                self,
                "Clear Multi-frame Annotations",
                f"Remove all {total_annotations} annotations from {num_frames} frames ({self.frame_range_start + 1}-{self.frame_range_end + 1})?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Clear all annotations in the range
                cleared_frames = 0
                for frame_idx in range(self.frame_range_start, self.frame_range_end + 1):
                    if frame_idx in self.annotations:
                        del self.annotations[frame_idx]
                        cleared_frames += 1
                
                # Update display
                self._update_annotation_list()
                self._update_statistics()
                self.statusBar().showMessage(f"Cleared {total_annotations} annotations from {cleared_frames} frames")
        else:
            # Single frame clearing (original behavior)
            if self.current_frame_index in self.annotations:
                count = len(self.annotations[self.current_frame_index])
                
                reply = QMessageBox.question(
                    self,
                    "Clear Annotations",
                    f"Remove all {count} annotations from current frame?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    del self.annotations[self.current_frame_index]
                    self._update_annotation_list()
                    self._update_statistics()
                    self.statusBar().showMessage(f"Cleared {count} annotations")
            else:
                QMessageBox.information(self, "No Annotations", "No annotations found in current frame.")
    
    def _clear_all_annotations(self):
        """Clear all annotations from all frames."""
        if not self.annotations:
            QMessageBox.information(self, "No Annotations", "No annotations found.")
            return
        
        # Count total annotations across all frames
        total_annotations = sum(len(frame_annotations) for frame_annotations in self.annotations.values())
        total_frames = len(self.annotations)
        
        reply = QMessageBox.question(
            self,
            "Clear All Annotations",
            f"Remove ALL {total_annotations} annotations from {total_frames} frames?\n\n"
            "This action cannot be undone!",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear all annotations
            self.annotations.clear()
            
            # Update display
            self._update_annotation_list()
            self._update_statistics()
            self.statusBar().showMessage(f"Cleared all {total_annotations} annotations from {total_frames} frames")
    
    def _quick_event(self, event_name: str):
        """Quick add event annotation."""
        self.event_input.setText(event_name)
        self._add_event_annotation()
    
    def _quick_contact_all(self, contact_state: bool):
        """Quick set all PSM contact states."""
        for checkbox in self.contact_checkboxes.values():
            checkbox.setChecked(contact_state)
        self._add_contact_annotation()
    
    def _quick_contact_single(self, psm: str, contact_state: bool):
        """Quick set contact state for a single PSM."""
        # Clear all checkboxes first
        for checkbox in self.contact_checkboxes.values():
            checkbox.setChecked(False)
        
        # Set only the specified PSM to contact state
        if psm in self.contact_checkboxes:
            self.contact_checkboxes[psm].setChecked(contact_state)
        
        self._add_contact_annotation()
    
    def select_save_folder(self):
        """Allow user to select a custom save folder."""
        folder_dialog = QFileDialog()
        selected_folder = folder_dialog.getExistingDirectory(
            self,
            "Select Save Folder",
            str(self.custom_save_folder or Path.home()),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if selected_folder:
            self.custom_save_folder = Path(selected_folder)
            # Update display with abbreviated path
            folder_str = str(self.custom_save_folder)
            if len(folder_str) > 50:
                folder_str = "..." + folder_str[-47:]
            self.save_folder_label.setText(f"Save to: {folder_str}")
            self.statusBar().showMessage(f"Save folder set to: {self.custom_save_folder}")
    
    def _update_video_display_size(self):
        """
        Update video display size based on config image max sizes and available GUI space.

        IMPROVEMENT: Smart adaptive sizing that:
        1. Calculates available space from window dimensions and splitter proportions
        2. Adapts to different image resolutions from config
        3. Minimizes scrollbar usage by utilizing available space efficiently
        4. Accounts for UI elements (menubar, statusbar, timeline, buttons, borders)
        """
        if self.config and hasattr(self, 'video_display_label'):
            try:
                # Get image size configuration
                size_config = self.config_loader.get_image_size_config()
                main_size = size_config.get('main_image_max_size')
                side_size = size_config.get('side_image_max_size')

                # Get GUI configuration for window dimensions
                gui_config = self.config_loader.get_gui_config()
                window_width = gui_config.get('window_width', 1910)
                window_height = gui_config.get('window_height', 1000)

                # Calculate combined image dimensions
                # Layout: [Left | Right] (top row)
                #         [Side1 | Side2] (bottom row)
                main_w, main_h = main_size[0], main_size[1]
                side_w, side_h = side_size[0], side_size[1]

                # Combined image size (what ImageProcessor creates)
                # Top row: main cameras side by side (width = main_w, already divided by 2 in combine_images)
                # Bottom row: side cameras side by side (width = 2*side_w)
                combined_image_width = max(main_w, 2 * side_w)
                combined_image_height = main_h + side_h

                # Calculate available space in video area
                # Splitter proportion: video area gets ~72% of width (1300/1800 from line 766)
                video_area_width_ratio = 0.72
                available_width = int(window_width * video_area_width_ratio)

                # Account for UI elements that take vertical space:
                # - Menubar: ~25px
                # - Status bar: ~25px
                # - Groupbox title/border: ~40px
                # - Camera buttons: ~60px
                # - Timeline controls: ~120px
                # Total: ~270px
                ui_vertical_overhead = 270
                available_height = window_height - ui_vertical_overhead

                # Calculate optimal display size with padding for scrollarea borders
                padding = 20  # Margins and borders
                max_display_width = min(combined_image_width, available_width - padding)
                max_display_height = min(combined_image_height, available_height - padding)

                # Ensure minimum size for usability (don't make it too small)
                min_width = 640  # Minimum reasonable width
                min_height = 480  # Minimum reasonable height
                display_width = max(max_display_width, min_width)
                display_height = max(max_display_height, min_height)

                # ADAPTIVE: If image is smaller than available space, use image size
                # If image is larger, use available space (scrollbar will appear)
                if combined_image_width < available_width:
                    display_width = combined_image_width
                if combined_image_height < available_height:
                    display_height = combined_image_height

                # Update video display size
                # Set both minimum and maximum to same value to create fixed size
                # This prevents the label from expanding/collapsing unexpectedly
                self.video_display_label.setMinimumSize(display_width, display_height)
                self.video_display_label.setMaximumSize(display_width, display_height)

                # Log sizing decision for debugging
                print(f"Video display sized: {display_width}x{display_height}")
                print(f"  Combined image: {combined_image_width}x{combined_image_height}")
                print(f"  Available space: {available_width}x{available_height}")
                print(f"  Scrollbar needed: Width={combined_image_width > available_width}, "
                      f"Height={combined_image_height > available_height}")

                self.statusBar().showMessage(
                    f"Video display: {display_width}x{display_height} "
                    f"(images: {main_size} main, {side_size} side)"
                )

            except Exception as e:
                print(f"Warning: Could not update video display size: {e}")
                print(f"Traceback: {traceback.format_exc()}")
    
    def get_save_folder(self) -> Path:
        """Get the current save folder (custom or from config)."""
        if self.custom_save_folder:
            return self.custom_save_folder
        elif self.config:
            return self.config_loader.get_save_folder()
        else:
            return Path.home() / "dvrk_annotations"
    
    def save_annotations(self):
        """
        Save annotations to files with optional backfilling of unlabeled frames.

        IMPROVED BEHAVIOR: Asks user whether to backfill unlabeled frames.
        - With backfill: Saves ALL frames for CONTACT ONLY (contact=0 for unlabeled)
                         Phase/Event only saved if explicitly annotated (sparse)
        - Without backfill: Only saves explicitly labeled frames

        CRITICAL: Backfilling ONLY applies to contact detection, NOT phase/event.
        This prevents generating meaningless default phase/event labels.
        """
        if not self.frame_files:
            QMessageBox.information(self, "No Data", "No frame data loaded. Please load a configuration first.")
            return

        try:
            # Calculate statistics about labeled vs unlabeled frames
            total_frames = len(self.frame_files)
            annotated_frames = len(self.annotations)
            unlabeled_frames = total_frames - annotated_frames

            # USER CHOICE: Ask whether to backfill unlabeled frames
            backfill_enabled = False

            if unlabeled_frames > 0:
                # Show detailed dialog explaining the choice
                dialog = QMessageBox(self)
                dialog.setIcon(QMessageBox.Question)
                dialog.setWindowTitle("Backfill Unlabeled Frames?")

                dialog_text = f"""<b>Labeling Status:</b><br>
• Total frames: {total_frames}<br>
• Labeled frames: {annotated_frames} ({annotated_frames/total_frames*100:.1f}%)<br>
• Unlabeled frames: {unlabeled_frames} ({unlabeled_frames/total_frames*100:.1f}%)<br>
<br>
<b>Do you want to backfill unlabeled frames with default contact values?</b><br>
<br>
<b>YES</b> - Save all {total_frames} frames:<br>
&nbsp;&nbsp;&nbsp;• Labeled frames: Use your annotations<br>
&nbsp;&nbsp;&nbsp;• Unlabeled frames: Use defaults for <b>CONTACT ONLY</b> (contact=0)<br>
&nbsp;&nbsp;&nbsp;• Phase/Event: Only saved if explicitly annotated (sparse)<br>
&nbsp;&nbsp;&nbsp;→ Recommended if labeling is complete<br>
<br>
<b>NO</b> - Save only {annotated_frames} labeled frames:<br>
&nbsp;&nbsp;&nbsp;• Only frames you explicitly annotated<br>
&nbsp;&nbsp;&nbsp;• No automatic data generation<br>
&nbsp;&nbsp;&nbsp;→ Recommended if labeling is still in progress<br>
"""
                dialog.setText(dialog_text)

                yes_btn = dialog.addButton("Yes, Backfill", QMessageBox.YesRole)
                no_btn = dialog.addButton("No, Only Labeled", QMessageBox.NoRole)
                cancel_btn = dialog.addButton("Cancel", QMessageBox.RejectRole)

                dialog.setDefaultButton(no_btn)  # Default to safer option
                dialog.exec_()

                clicked_button = dialog.clickedButton()

                if clicked_button == cancel_btn:
                    return  # User cancelled
                elif clicked_button == yes_btn:
                    backfill_enabled = True
                    self.statusBar().showMessage(f"Saving with backfill: {total_frames} frames total")
                else:  # no_btn
                    backfill_enabled = False
                    self.statusBar().showMessage(f"Saving without backfill: {annotated_frames} frames only")
            else:
                # All frames are labeled, no need to ask
                backfill_enabled = True
                self.statusBar().showMessage("All frames labeled - saving all")

            # Get save folder (custom or from config)
            base_folder = self.get_save_folder()
            categories_saved = set()

            # Determine which frames to save based on user choice
            if backfill_enabled:
                # Save ALL frames (labeled + backfilled)
                frames_to_save = range(total_frames)
            else:
                # Save only explicitly labeled frames
                frames_to_save = sorted(self.annotations.keys())

            for frame_idx in frames_to_save:
                # Get annotations for this frame (empty list if not annotated)
                frame_annotations = self.annotations.get(frame_idx, [])
                # Separate contact, phase, and other annotations
                contact_annotations = [a for a in frame_annotations if a["category"].startswith("contact_")]
                phase_annotations = [a for a in frame_annotations if a["category"].startswith("phase_")]
                other_annotations = [a for a in frame_annotations if not (
                    a["category"].startswith("contact_") or a["category"].startswith("phase_")
                )]
                
                # IMPORTANT FIX: Always save contact detection labels for ALL frames
                # This ensures both contact and non-contact frames have labels
                # Initialize all active PSMs to 0 (no contact by default)
                combined_contact_data = {}
                for psm in self.active_psms:
                    combined_contact_data[psm] = 0

                # Set contacts to 1 for PSMs that have contact annotations
                for annotation in contact_annotations:
                    category = annotation["category"]
                    data = annotation["data"]
                    psm = category.replace("contact_", "")
                    if psm in data and data[psm]:
                        combined_contact_data[psm] = 1

                # Save contact detection data for this frame (always, even if no contact)
                contact_folder = base_folder / "annotation" / "contact_detection"
                if not contact_folder.exists():
                    create_folder(contact_folder)

                frame_file = contact_folder / f"{frame_idx}.json"
                with open(frame_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_contact_data, f, indent=2, ensure_ascii=False)

                categories_saved.add("contact_detection")

                # CRITICAL FIX: Only save phase labels for frames that have phase annotations
                # Do NOT backfill phase with null values - phase is sparse (only when annotated)
                # This ensures we only save phase data when user explicitly added it
                if phase_annotations:
                    # Initialize phase data structure
                    combined_phase_data = {"phase": {}}
                    for psm in self.active_psms:
                        combined_phase_data["phase"][psm] = None

                    # Set phases for PSMs that have phase annotations
                    for annotation in phase_annotations:
                        category = annotation["category"]
                        data = annotation["data"]
                        psm = category.replace("phase_", "")
                        if "phase" in data and psm in data["phase"]:
                            combined_phase_data["phase"][psm] = data["phase"][psm]

                    # Save phase data ONLY if this frame has phase annotations
                    phase_folder = base_folder / "annotation" / "phase"
                    if not phase_folder.exists():
                        create_folder(phase_folder)

                    frame_file = phase_folder / f"{frame_idx}.json"
                    with open(frame_file, 'w', encoding='utf-8') as f:
                        json.dump(combined_phase_data, f, indent=2, ensure_ascii=False)

                    categories_saved.add("phase")
                
                # Group other annotations by category to handle multiple events properly
                category_annotations = {}
                for annotation in other_annotations:
                    category = annotation["category"]
                    data = annotation["data"]
                    
                    # Skip individual PSM categories - they're already handled as combined files above
                    if category.startswith("contact_") or category.startswith("phase_"):
                        continue
                    
                    if category not in category_annotations:
                        category_annotations[category] = []
                    category_annotations[category].append(data)
                
                # Save annotations by category, handling multiple events properly
                for category, annotation_data_list in category_annotations.items():
                    annotation_folder = base_folder / "annotation" / category
                    if not annotation_folder.exists():
                        create_folder(annotation_folder)
                    
                    frame_file = annotation_folder / f"{frame_idx}.json"
                    
                    # For event category, save multiple annotations as an array or object with events array
                    if category == "event":
                        # Create a structure that includes all event labels for this frame
                        if len(annotation_data_list) == 1:
                            # Single event - maintain backward compatibility with simple structure
                            event_data = annotation_data_list[0]
                        else:
                            # Multiple events - create an array structure to include all events
                            event_labels = []
                            for event_data in annotation_data_list:
                                if isinstance(event_data, dict) and "event" in event_data:
                                    event_labels.append(event_data["event"])
                                elif isinstance(event_data, str):
                                    event_labels.append(event_data)
                            
                            # Save as an object with events array to include all multiple event labels
                            event_data = {"events": event_labels}
                        
                        with open(frame_file, 'w', encoding='utf-8') as f:
                            json.dump(event_data, f, indent=2, ensure_ascii=False)
                    else:
                        # For non-event categories, save the last annotation (original behavior)
                        with open(frame_file, 'w', encoding='utf-8') as f:
                            json.dump(annotation_data_list[-1], f, indent=2, ensure_ascii=False)
                    
                    categories_saved.add(category)
            
            total_annotations = sum(len(annotations) for annotations in self.annotations.values())
            annotated_frames_count = len(self.annotations)
            frames_saved = len(frames_to_save)

            # Update status message based on backfill choice
            if backfill_enabled:
                self.statusBar().showMessage(
                    f"Saved {frames_saved} frames WITH backfill ({annotated_frames_count} annotated, "
                    f"{frames_saved - annotated_frames_count} backfilled with defaults)"
                )

                QMessageBox.information(
                    self,
                    "Save Complete (With Backfill)",
                    f"✓ Saved labels for ALL {frames_saved} frames\n"
                    f"  • {annotated_frames_count} frames with your annotations\n"
                    f"  • {frames_saved - annotated_frames_count} frames backfilled with defaults\n"
                    f"    (CONTACT ONLY: contact=0 for unannotated frames)\n"
                    f"  • Phase/Event: Only saved if explicitly annotated (sparse)\n"
                    f"  • Categories: {', '.join(sorted(categories_saved))}\n\n"
                    f"Location: {base_folder / 'annotation'}\n\n"
                    f"✓ Contact label count = Frame count ({frames_saved} files)"
                )
            else:
                self.statusBar().showMessage(
                    f"Saved {frames_saved} frames WITHOUT backfill (only explicitly labeled frames)"
                )

                QMessageBox.information(
                    self,
                    "Save Complete (Without Backfill)",
                    f"✓ Saved labels for {frames_saved} explicitly labeled frames\n"
                    f"  • {annotated_frames_count} frames with annotations\n"
                    f"  • {total_frames - frames_saved} frames NOT saved (unlabeled)\n"
                    f"  • Categories: {', '.join(sorted(categories_saved))}\n\n"
                    f"Location: {base_folder / 'annotation'}\n\n"
                    f"NOTE: {total_frames - frames_saved} frames still need labeling.\n"
                    f"Save again with backfill when labeling is complete."
                )
            
        except Exception as e:
            error_msg = f"Error saving annotations: {str(e)}"
            print(f"Save error: {traceback.format_exc()}")
            QMessageBox.critical(self, "Save Error", error_msg)
    
    def export_annotations(self):
        """Export all annotations to a single file."""
        if not self.annotations:
            QMessageBox.information(self, "No Data", "No annotations to export.")
            return
        
        export_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Annotations",
            "dvrk_annotations_export.json",
            "JSON files (*.json);;All files (*)"
        )
        
        if export_path:
            try:
                export_data = {
                    "metadata": {
                        "total_frames": len(self.frame_files),
                        "annotated_frames": len(self.annotations),
                        "export_timestamp": datetime.datetime.now().isoformat(),
                        "active_psms": self.active_psms,
                        "categories": ["event", "phase", "contact_detection"]
                    },
                    "annotations": self.annotations
                }
                
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Export Complete", f"Annotations exported to:\n{export_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Error exporting: {str(e)}")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
        dVRK Data Annotation Tool
        
        Comprehensive GUI for annotating surgical events, phases, and contact detection
        in dVRK (da Vinci Research Kit) multimodal data.
        
        Features:
        • Multi-camera video display with proper sizing constraints
        • Event, phase, and contact annotation capabilities  
        • Multi-frame batch labeling functionality
        • Auto-play with configurable speed (1x = 30 Hz)
        • PSM contact detection with boolean labels
        • Proper folder structure for annotations
        • No auto-save, manual control over saving
        
        Built with PyQt5 and integrated with Hydra configuration system.
        """
        
        QMessageBox.about(self, "About dVRK Data Annotation Tool", about_text)
    
    def closeEvent(self, event):
        """Handle application closing."""
        # Ask about unsaved changes
        if self.annotations:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved annotations. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.save_annotations()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        
        # Stop timers and threads
        if self.is_playing:
            self.auto_play_timer.stop()
        
        if self.image_processor:
            self.image_processor.stop()
            self.image_processor.wait()
        
        event.accept()


def main():
    """Main application entry point."""
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("dVRK Data Annotation Tool")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    try:
        window = DataAnnotationGUI()
        window.show()
        
        print("dVRK Data Annotation Tool started")
        print("Load configuration to begin annotation")
        
        # Start event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Application error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Show error to user
        QMessageBox.critical(None, "Application Error", 
                           f"Failed to start application:\n{str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()