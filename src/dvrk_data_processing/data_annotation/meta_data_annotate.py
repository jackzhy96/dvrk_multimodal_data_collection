"""
dVRK Meta Data Annotation GUI

A simplified PyQt-based GUI for annotating surgical metadata in dVRK multi-modal data.
This tool provides an intuitive interface for:

- Multi-camera video display with synchronized timeline navigation
- Metadata annotation (user ID, operator skill level, data type)
- Failure and recovery frame labeling
- Multi-frame batch labeling functionality
- Auto-play with configurable speed
- Metadata saving to single JSON file
"""

import json
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

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


# Reuse ConfigLoader and ImageProcessor from data_annotate.py
# (Import them directly to avoid code duplication)
try:
    from dvrk_data_processing.data_annotation.data_annotate import ConfigLoader, ImageProcessor
except ImportError:
    # If import fails, define minimal versions here
    class ConfigLoader:
        """Loads and manages configuration using Hydra config system."""

        def __init__(self):
            self.config: Optional[DictConfig] = None
            self.config_path: Optional[Path] = None

        def load_config(self, config_path: Optional[Path] = None) -> DictConfig:
            """Load configuration using Hydra config system."""
            try:
                if config_path is None:
                    config_path = self._find_config_file()

                if not config_path or not config_path.exists():
                    raise FileNotFoundError(f"Configuration file not found: {config_path}")

                self.config_path = config_path

                with hydra.initialize_config_dir(config_dir=str(config_path.parent)):
                    cfg = hydra.compose(config_name=config_path.stem)
                    self.config = cfg
                    OmegaConf.resolve(self.config)
                    return self.config

            except Exception as e:
                print(f"Error loading configuration: {e}")
                raise

        def _find_config_file(self) -> Optional[Path]:
            """Search for config_annotation.yaml file in the project."""
            search_paths = [
                Path(__file__).resolve().parent.parent.parent.parent / "config",
                Path(__file__).resolve().parent / "config",
            ]

            for search_path in search_paths:
                if search_path.exists():
                    config_file = search_path / "config_annotation.yaml"
                    if config_file.exists():
                        return config_file

            return None

        def get_image_paths(self) -> Dict[str, Path]:
            """Get image paths for all cameras based on configuration."""
            if not self.config:
                raise ValueError("Configuration not loaded")

            image_paths = {}

            if hasattr(self.config, 'left_image_path') and self.config.left_image_path:
                image_paths['left'] = Path(self.config.left_image_path)

            if hasattr(self.config, 'right_image_path') and self.config.right_image_path:
                image_paths['right'] = Path(self.config.right_image_path)

            for i in range(1, 3):
                attr_name = f'side_camera_{i}_path'
                if hasattr(self.config, attr_name):
                    side_path = getattr(self.config, attr_name)
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

            main_size = getattr(self.config, 'main_image_max_size')
            side_size = getattr(self.config, 'side_image_max_size')

            def convert_to_list(value, default):
                if value is None:
                    return default if default is not None else [854, 480]
                elif hasattr(value, '_content'):
                    return OmegaConf.to_container(value)
                elif isinstance(value, (list, tuple)):
                    return list(value)
                else:
                    return default if default is not None else [854, 480]

            return {
                'resize_max_size': getattr(self.config, 'resize_max_size', False),
                'main_image_max_size': convert_to_list(main_size, [854, 480]),
                'side_image_max_size': convert_to_list(side_size, [640, 480])
            }

        def get_gui_config(self) -> Dict[str, Any]:
            """Get GUI configuration parameters for better scalability."""
            if not self.config or not hasattr(self.config, 'gui_config'):
                return {
                    'window_width': 1910,
                    'window_height': 1000,
                    'default_playback_speed_ms': 33,
                    'min_playback_speed_ms': 10,
                    'max_playback_speed_ms': 1000,
                    'image_loader_refresh_ms': 50
                }

            gui_config = self.config.gui_config
            return {
                'window_width': getattr(gui_config, 'window_width', 1910),
                'window_height': getattr(gui_config, 'window_height', 1000),
                'default_playback_speed_ms': getattr(gui_config, 'default_playback_speed_ms', 33),
                'min_playback_speed_ms': getattr(gui_config, 'min_playback_speed_ms', 10),
                'max_playback_speed_ms': getattr(gui_config, 'max_playback_speed_ms', 1000),
                'image_loader_refresh_ms': getattr(gui_config, 'image_loader_refresh_ms', 50)
            }

    class ImageProcessor(QThread):
        """Background thread for loading and processing video frames."""

        images_loaded = pyqtSignal(np.ndarray, str)
        loading_error = pyqtSignal(str)

        def __init__(self, image_paths: Dict[str, Path], frame_files: List[Path],
                     size_config: Dict[str, Any], gui_config: Optional[Dict[str, Any]] = None):
            super().__init__()
            self.image_paths = image_paths
            self.frame_files = frame_files
            self.size_config = size_config
            self.gui_config = gui_config if gui_config else {'image_loader_refresh_ms': 50}
            self.current_frame_index = 0
            self.target_frame_index = 0
            self.running = True

            self.resize_max_size = size_config.get('resize_max_size', False)
            self.main_image_max_size = size_config.get('main_image_max_size', [854, 480])
            self.side_image_max_size = size_config.get('side_image_max_size', [640, 480])

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

                refresh_ms = self.gui_config.get('image_loader_refresh_ms', 50)
                self.msleep(refresh_ms)

        def load_frame(self, frame_index: int):
            """Load and process images for specified frame with proper sizing."""
            if frame_index >= len(self.frame_files):
                return

            frame_file = self.frame_files[frame_index]
            frame_stem = frame_file.stem

            loaded_images = {}

            for camera_name, camera_path in self.image_paths.items():
                if not camera_path.exists():
                    continue

                image_file = camera_path / f"{frame_stem}{frame_file.suffix}"
                if image_file.exists():
                    img = cv2.imread(str(image_file))
                    if img is not None:
                        resized_img = self._resize_image(img, camera_name)
                        loaded_images[camera_name] = resized_img

            if not loaded_images:
                self.loading_error.emit(f"No images found for frame {frame_index}")
                return

            combined_image = self._combine_images(loaded_images)
            camera_list = ", ".join(loaded_images.keys())
            frame_info = f"Frame {frame_index + 1}/{len(self.frame_files)} | {frame_stem} | Cameras: {camera_list}"

            self.images_loaded.emit(combined_image, frame_info)

        def _resize_image(self, image: np.ndarray, camera_name: str) -> np.ndarray:
            """Resize image according to configuration and camera type."""
            if camera_name in ['left', 'right']:
                max_size = self.main_image_max_size
            else:
                max_size = self.side_image_max_size

            h, w = image.shape[:2]
            max_w, max_h = max_size[0], max_size[1]

            needs_resize = self.resize_max_size or (w > max_w) or (h > max_h)

            if not needs_resize:
                return image

            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized

        def _combine_images(self, images: Dict[str, np.ndarray]) -> np.ndarray:
            """Combine multiple camera images for display."""
            if len(images) == 1:
                return list(images.values())[0]

            left_img = images.get('left')
            right_img = images.get('right')
            side1_img = images.get('side_1')
            side2_img = images.get('side_2')

            if left_img is not None and right_img is not None and side1_img is None and side2_img is None:
                stereo_images = [left_img, right_img]
                max_h = max(img.shape[0] for img in stereo_images)
                resized = []
                for img in stereo_images:
                    if img.shape[0] != max_h:
                        scale = max_h / img.shape[0]
                        new_w = int(img.shape[1] * scale)
                        resized.append(cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA))
                    else:
                        resized.append(img)
                return cv2.hconcat(resized)

            # For 2x2 grid layout
            main_height = self.main_image_max_size[1]
            main_width = self.main_image_max_size[0] // 2
            side_height = self.side_image_max_size[1]
            side_width = self.side_image_max_size[0]

            left_resized = cv2.resize(left_img, (main_width, main_height), interpolation=cv2.INTER_AREA) if left_img is not None else np.zeros((main_height, main_width, 3), dtype=np.uint8)
            right_resized = cv2.resize(right_img, (main_width, main_height), interpolation=cv2.INTER_AREA) if right_img is not None else np.zeros((main_height, main_width, 3), dtype=np.uint8)
            side1_resized = cv2.resize(side1_img, (side_width, side_height), interpolation=cv2.INTER_AREA) if side1_img is not None else np.zeros((side_height, side_width, 3), dtype=np.uint8)
            side2_resized = cv2.resize(side2_img, (side_width, side_height), interpolation=cv2.INTER_AREA) if side2_img is not None else np.zeros((side_height, side_width, 3), dtype=np.uint8)

            top_row = cv2.hconcat([left_resized, right_resized])
            bottom_row = cv2.hconcat([side1_resized, side2_resized])

            max_width = max(top_row.shape[1], bottom_row.shape[1])
            if top_row.shape[1] < max_width:
                padding = np.zeros((top_row.shape[0], max_width - top_row.shape[1], 3), dtype=np.uint8)
                top_row = cv2.hconcat([top_row, padding])
            if bottom_row.shape[1] < max_width:
                padding = np.zeros((bottom_row.shape[0], max_width - bottom_row.shape[1], 3), dtype=np.uint8)
                bottom_row = cv2.hconcat([bottom_row, padding])

            combined = cv2.vconcat([top_row, bottom_row])
            return combined

        def stop(self):
            """Stop the image processor thread."""
            self.running = False


class MetaDataAnnotationGUI(QMainWindow):
    """
    Main GUI class for metadata annotation.

    Provides a simplified interface for annotating surgical metadata:
    - User ID, operator skill level, data type
    - Failure and recovery frame labels
    - Multi-frame batch labeling
    - Auto-play functionality with configurable speed
    - Metadata saving to single JSON file
    """

    def __init__(self):
        super().__init__()

        # Configuration and data management
        self.config_loader = ConfigLoader()
        self.config: Optional[DictConfig] = None
        self.image_paths: Dict[str, Path] = {}
        self.frame_files: List[Path] = []
        self.custom_save_folder: Optional[Path] = None

        # Metadata storage
        self.metadata = {
            'user_id': '',
            'operator_skill_level': '',
            'case_type': '',
            'tool': {},  # Dictionary of PSM tools, e.g., {'PSM1': 'Large_Needle_Driver', 'PSM2': 'Prograsp_Forceps'}
            'failure': [],  # List of tuples (start_idx, end_idx)
            'recovery': []  # List of tuples (start_idx, end_idx)
        }

        # GUI state
        self.current_frame_index = 0
        self.image_processor: Optional[ImageProcessor] = None
        self.has_unsaved_changes = False  # Track unsaved modifications

        # Auto-play state (EXACT same as data_annotate.py)
        self.is_playing = False
        self.auto_play_timer = QTimer()
        self.auto_play_timer.timeout.connect(self._auto_play_step)
        self.playback_speed_ms = 33  # Default 30 Hz

        # Multi-frame labeling
        self.multi_frame_mode = False
        self.frame_range_start: Optional[int] = None
        self.frame_range_end: Optional[int] = None

        # GUI components (initialized in setup methods)
        self.video_display_label: Optional[QLabel] = None
        self.timeline_slider: Optional[QSlider] = None
        self.frame_info_label: Optional[QLabel] = None
        self.annotation_list: Optional[QListWidget] = None

        # Metadata annotation controls
        self.user_id_input: Optional[QLineEdit] = None
        self.skill_level_dropdown: Optional[QComboBox] = None
        self.skill_level_custom_input: Optional[QLineEdit] = None
        self.case_type_dropdown: Optional[QComboBox] = None
        self.case_type_custom_input: Optional[QLineEdit] = None

        # PSM Tool annotation controls - dynamically created based on config
        self.psm_tool_dropdowns: Dict[str, QComboBox] = {}  # e.g., {'PSM1': QComboBox, 'PSM2': QComboBox}
        self.psm_tool_custom_inputs: Dict[str, QLineEdit] = {}  # e.g., {'PSM1': QLineEdit, 'PSM2': QLineEdit}


        # Multi-frame controls
        self.multi_frame_checkbox: Optional[QCheckBox] = None
        self.range_display_label: Optional[QLabel] = None

        # Camera button controls for opening original images (EXACT same as data_annotate.py)
        self.camera_buttons: Dict[str, QPushButton] = {}
        self.left_camera_btn: Optional[QPushButton] = None
        self.right_camera_btn: Optional[QPushButton] = None
        self.side1_camera_btn: Optional[QPushButton] = None
        self.side2_camera_btn: Optional[QPushButton] = None

        # Independent image windows tracking (for separate, non-modal windows) (EXACT same as data_annotate.py)
        self._image_windows: List = []

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        # Set window properties (EXACT same as data_annotate.py)
        self.setWindowTitle("dVRK Meta Data Annotation Tool")

        # Get GUI configuration parameters - using configurable values for better scalability
        gui_config = self.config_loader.get_gui_config() if hasattr(self, 'config_loader') else {
            'window_width': 1910, 'window_height': 1000
        }

        # Set window geometry using configurable parameters (EXACT same as data_annotate.py)
        # Position at (5, 10) to minimize top edge while ensuring full bottom visibility
        # Height set to 1000px (safe maximum for 1080P with taskbar + decorations)
        # This ensures NO parts of GUI are cut off at bottom
        self.setGeometry(5, 10, gui_config['window_width'], gui_config['window_height'])

        # Apply global stylesheet (EXACT same as data_annotate.py)
        self.setStyleSheet(self._get_app_stylesheet())

        # Create central widget and main layout (EXACT same as data_annotate.py)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create horizontal splitter for video and controls (EXACT same as data_annotate.py)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left side: Video display area (larger for better viewing)
        video_widget = self._create_video_widget()
        main_splitter.addWidget(video_widget)

        # Right side: Control panel
        control_widget = self._create_control_widget()
        main_splitter.addWidget(control_widget)

        # Set splitter proportions optimized for 1920x1080 (EXACT same as data_annotate.py)
        main_splitter.setSizes([1080, 600])

        # Create status bar and menu
        self.statusBar().showMessage("Ready - Load configuration to begin")
        self._create_menu_bar()

    def _get_app_stylesheet(self) -> str:
        """Get the application stylesheet (EXACT same as data_annotate.py)."""
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
        """Create application menu bar (EXACT same as data_annotate.py)."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        load_config_action = file_menu.addAction('Load Configuration')
        load_config_action.triggered.connect(self.load_configuration)

        file_menu.addSeparator()

        save_action = file_menu.addAction('Save Meta Data')
        save_action.triggered.connect(self.save_metadata)

        file_menu.addSeparator()

        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)

        # Help menu
        help_menu = menubar.addMenu('Help')
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About dVRK Meta Data Annotation Tool",
            "dVRK Meta Data Annotation Tool\n\n"
            "A GUI for annotating surgical metadata in dVRK multi-modal data.\n\n"
            "Features:\n"
            "- Multi-camera video display\n"
            "- Metadata annotation (user ID, skill level, data type)\n"
            "- Failure and recovery frame labeling\n"
            "- Multi-frame batch labeling\n\n"
            "Version 1.0"
        )

    def _create_video_widget(self) -> QWidget:
        """Create the video display widget (EXACT same as data_annotate.py)."""
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
        """Create timeline control widget with auto-play functionality (EXACT same as data_annotate.py)."""
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

        # Auto-play controls with speed configuration
        playback_layout = QHBoxLayout()

        # Jump to start/end
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

        # Speed control with dropdown menu
        playback_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        speed_options = [
            ("0.1x", 333),
            ("0.25x", 133),
            ("0.5x", 67),
            ("1x [30.00 Hz]", 33),
            ("2x", 17),
            ("4x", 8),
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
        """Create buttons for opening original images in new windows (EXACT same as data_annotate.py)."""
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
        self.left_camera_btn.setEnabled(False)
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
        """Create the control panel widget with 2-column layout."""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Multi-frame labeling controls span both columns (top of controls)
        multi_frame_group = self._create_multi_frame_section()
        control_layout.addWidget(multi_frame_group)

        # Create scrollable area for the 2-column layout
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Create 2-column layout for the main controls
        columns_layout = QHBoxLayout()

        # Left column: Configuration, Failure Annotation, Recovery Annotation, Meta Data Annotation
        left_column = QVBoxLayout()
        left_column_widget = QWidget()
        left_column_widget.setLayout(left_column)

        config_group = self._create_config_section()
        left_column.addWidget(config_group)

        failure_group = self._create_failure_section()
        left_column.addWidget(failure_group)

        recovery_group = self._create_recovery_section()
        left_column.addWidget(recovery_group)

        metadata_group = self._create_metadata_section()
        left_column.addWidget(metadata_group)

        # Add stretch to push content to top of left column
        left_column.addStretch()

        # Right column: Current Annotations, Statistics, Save
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

        # Add columns layout to scroll layout
        scroll_layout.addLayout(columns_layout)

        scroll_area.setWidget(scroll_widget)
        control_layout.addWidget(scroll_area)

        return control_widget

    def _create_config_section(self) -> QGroupBox:
        """Create configuration section."""
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)

        load_config_btn = QPushButton("Load Configuration")
        load_config_btn.clicked.connect(self.load_configuration)
        config_layout.addWidget(load_config_btn)

        load_metadata_btn = QPushButton("Load Meta Data File")
        load_metadata_btn.clicked.connect(self.load_metadata_file)
        config_layout.addWidget(load_metadata_btn)

        return config_group

    def _create_failure_section(self) -> QGroupBox:
        """Create failure annotation section using multi-frame labeling."""
        failure_group = QGroupBox("Failure Annotation")
        failure_layout = QVBoxLayout(failure_group)

        # Button to add failure label using multi-frame range
        add_failure_btn = QPushButton("Add Failure Label")
        add_failure_btn.clicked.connect(self._add_failure_label)
        failure_layout.addWidget(add_failure_btn)

        # Button to clear all failure labels
        clear_failure_btn = QPushButton("Clear All")
        clear_failure_btn.clicked.connect(self._clear_failure_labels)
        failure_layout.addWidget(clear_failure_btn)

        return failure_group

    def _create_recovery_section(self) -> QGroupBox:
        """Create recovery annotation section using multi-frame labeling."""
        recovery_group = QGroupBox("Recovery Annotation")
        recovery_layout = QVBoxLayout(recovery_group)

        # Button to add recovery label using multi-frame range
        add_recovery_btn = QPushButton("Add Recovery Label")
        add_recovery_btn.clicked.connect(self._add_recovery_label)
        recovery_layout.addWidget(add_recovery_btn)

        # Button to clear all recovery labels
        clear_recovery_btn = QPushButton("Clear All")
        clear_recovery_btn.clicked.connect(self._clear_recovery_labels)
        recovery_layout.addWidget(clear_recovery_btn)

        return recovery_group

    def _create_multi_frame_section(self) -> QGroupBox:
        """Create multi-frame labeling section (similar to data_annotate.py)."""
        multi_frame_group = QGroupBox("Multi-Frame Labeling")
        multi_frame_layout = QVBoxLayout(multi_frame_group)

        # Enable multi-frame mode
        self.multi_frame_checkbox = QCheckBox("Enable multi-frame labeling")
        self.multi_frame_checkbox.stateChanged.connect(self._toggle_multi_frame_mode)
        multi_frame_layout.addWidget(self.multi_frame_checkbox)

        # Range selection controls in horizontal layout
        range_layout = QHBoxLayout()

        self.set_start_btn = QPushButton("Set Start")
        self.set_start_btn.clicked.connect(self._set_start_frame)
        self.set_start_btn.setEnabled(False)
        range_layout.addWidget(self.set_start_btn)

        self.set_end_btn = QPushButton("Set End")
        self.set_end_btn.clicked.connect(self._set_end_frame)
        self.set_end_btn.setEnabled(False)
        range_layout.addWidget(self.set_end_btn)

        multi_frame_layout.addLayout(range_layout)

        # Range display
        self.range_display_label = QLabel("Range: Not set")
        self.range_display_label.setStyleSheet("font-style: italic;")
        multi_frame_layout.addWidget(self.range_display_label)

        return multi_frame_group

    def _create_metadata_section(self) -> QGroupBox:
        """Create metadata annotation section."""
        metadata_group = QGroupBox("Meta Data Annotation")
        metadata_layout = QFormLayout(metadata_group)

        # User ID input
        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText("Enter user ID...")
        self.user_id_input.textChanged.connect(lambda: self._mark_unsaved_changes())
        metadata_layout.addRow("User ID:", self.user_id_input)

        # Operator Skill Level dropdown (split into two lines to save space)
        self.skill_level_dropdown = QComboBox()
        self.skill_level_dropdown.addItems(["Expert", "Intermediate", "Novice", "Others"])
        self.skill_level_dropdown.currentTextChanged.connect(self._on_skill_level_changed)
        self.skill_level_dropdown.currentTextChanged.connect(lambda: self._mark_unsaved_changes())
        metadata_layout.addRow("Operator\nSkill Level:", self.skill_level_dropdown)

        # Custom skill level input (hidden by default)
        self.skill_level_custom_input = QLineEdit()
        self.skill_level_custom_input.setPlaceholderText("Enter custom skill level...")
        self.skill_level_custom_input.textChanged.connect(lambda: self._mark_unsaved_changes())
        self.skill_level_custom_input.setVisible(False)
        metadata_layout.addRow("", self.skill_level_custom_input)

        # Case Type dropdown
        self.case_type_dropdown = QComboBox()
        self.case_type_dropdown.addItems([
            "Clinical", "Ex-vivo", "Table-Top Phantom",
            "Digital Simulation", "Physical Simulation", "Others"
        ])
        self.case_type_dropdown.currentTextChanged.connect(self._on_case_type_changed)
        self.case_type_dropdown.currentTextChanged.connect(lambda: self._mark_unsaved_changes())
        metadata_layout.addRow("Case Type:", self.case_type_dropdown)

        # Custom case type input (hidden by default)
        self.case_type_custom_input = QLineEdit()
        self.case_type_custom_input.setPlaceholderText("Enter custom case type...")
        self.case_type_custom_input.textChanged.connect(lambda: self._mark_unsaved_changes())
        self.case_type_custom_input.setVisible(False)
        metadata_layout.addRow("", self.case_type_custom_input)

        # PSM Tool dropdowns - will be populated after configuration is loaded
        # We create placeholders here and populate them in load_configuration()
        self.psm_tool_section_start_index = metadata_layout.rowCount()

        return metadata_group

    def _create_current_annotations_section(self) -> QGroupBox:
        """Create current annotations display section with management buttons."""
        current_group = QGroupBox("Current Annotations")
        current_layout = QVBoxLayout(current_group)

        self.annotation_list = QListWidget()
        self.annotation_list.setMaximumHeight(150)
        current_layout.addWidget(self.annotation_list)

        # Annotation management buttons (similar to data_annotate.py)
        annotation_btn_layout = QHBoxLayout()

        self.remove_selected_btn = QPushButton("Remove\nSelected")
        self.remove_selected_btn.clicked.connect(self._remove_selected_annotation)
        self.remove_selected_btn.setStyleSheet("background-color: #f44336; color: white;")
        annotation_btn_layout.addWidget(self.remove_selected_btn)

        # Add "Clear All" button to clear all failure and recovery labels
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

        # Add reset/reload button for statistics (similar to data_annotate.py)
        reset_stats_btn = QPushButton("Reset/Reload Stats")
        reset_stats_btn.clicked.connect(self._reset_statistics)
        reset_stats_btn.setStyleSheet("font-size: 12px; padding: 5px;")
        stats_layout.addWidget(reset_stats_btn)

        return stats_group

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

        # Button to save metadata to selected folder
        save_btn = QPushButton("Save Meta Data")
        save_btn.clicked.connect(self.save_metadata)
        save_layout.addWidget(save_btn)

        return save_group

    # Event handlers
    def _on_skill_level_changed(self, text: str):
        """Handle skill level dropdown change."""
        self.skill_level_custom_input.setVisible(text == "Others")

    def _on_case_type_changed(self, text: str):
        """Handle case type dropdown change."""
        self.case_type_custom_input.setVisible(text == "Others")

    def _on_psm_tool_changed(self, psm_name: str, text: str):
        """Handle PSM tool dropdown change."""
        if psm_name in self.psm_tool_custom_inputs:
            self.psm_tool_custom_inputs[psm_name].setVisible(text == "Others")
        self._mark_unsaved_changes()

    def _mark_unsaved_changes(self):
        """Mark that there are unsaved changes."""
        self.has_unsaved_changes = True

    def _clear_psm_tool_controls(self):
        """
        Safely clear all existing PSM tool controls from the layout.

        This method removes PSM tool dropdowns and custom inputs from the
        QFormLayout and cleans up the tracking dictionaries. It's safe to
        call multiple times and handles cases where widgets may not exist.
        """
        # Get the metadata group's layout (FormLayout)
        metadata_group = None
        for widget in self.findChildren(QGroupBox):
            if widget.title() == "Meta Data Annotation":
                metadata_group = widget
                break

        if not metadata_group:
            # Clear dictionaries even if we can't find the layout
            self.psm_tool_dropdowns.clear()
            self.psm_tool_custom_inputs.clear()
            return

        metadata_layout = metadata_group.layout()
        if not isinstance(metadata_layout, QFormLayout):
            # Clear dictionaries even if layout is wrong type
            self.psm_tool_dropdowns.clear()
            self.psm_tool_custom_inputs.clear()
            return

        # First, collect all widgets that need to be removed to avoid reference issues
        widgets_to_remove = []

        # Iterate backwards to safely identify PSM-related rows
        i = metadata_layout.rowCount() - 1
        while i >= 0:
            # Get the label item for this row
            label_item = metadata_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = metadata_layout.itemAt(i, QFormLayout.FieldRole)

            should_remove = False

            if label_item and label_item.widget():
                label_text = label_item.widget().text()
                # Check if this row is for a PSM tool (e.g., "PSM1 Tool:", "PSM2 Tool:")
                if label_text and ("PSM" in label_text) and ("Tool" in label_text):
                    should_remove = True
                # Also check for rows with empty labels that contain PSM custom input widgets
                elif label_text == "" and field_item and field_item.widget():
                    # Check if this is a PSM tool custom input by checking if it's in our tracking dict
                    field_widget = field_item.widget()
                    if isinstance(field_widget, QLineEdit):
                        # Check if this widget is one of our PSM tool custom inputs
                        for psm_custom_input in self.psm_tool_custom_inputs.values():
                            if field_widget is psm_custom_input:
                                should_remove = True
                                break

            if should_remove:
                widgets_to_remove.append(i)

            i -= 1

        # Now remove all identified rows
        for row_index in widgets_to_remove:
            metadata_layout.removeRow(row_index)

        # Now safely clear the dictionaries
        # Don't try to access the widgets, just clear the references
        self.psm_tool_dropdowns.clear()
        self.psm_tool_custom_inputs.clear()

    def _create_psm_tool_controls(self):
        """
        Create PSM tool dropdown controls dynamically based on configuration.

        This method is called after configuration is loaded to add tool selection
        dropdowns for each enabled PSM (PSM1, PSM2, PSM3).
        """
        if not self.config:
            return

        # First, safely clear any existing PSM tool controls
        self._clear_psm_tool_controls()

        # Get the metadata group's layout (FormLayout)
        metadata_group = None
        for widget in self.findChildren(QGroupBox):
            if widget.title() == "Meta Data Annotation":
                metadata_group = widget
                break

        if not metadata_group:
            return

        metadata_layout = metadata_group.layout()
        if not isinstance(metadata_layout, QFormLayout):
            return

        # Define tool options
        tool_options = [
            "Large_Needle_Driver",
            "Prograsp_Forceps",
            "Maryland_Bipolar_Forceps",
            "Curved_Scissors",
            "Others"
        ]

        # Check which PSMs are enabled and create dropdowns for each
        for psm_num in [1, 2, 3]:
            psm_name = f"PSM{psm_num}"
            enable_attr = f"enable_{psm_name}"

            # Check if this PSM is enabled in config
            if hasattr(self.config, enable_attr) and getattr(self.config, enable_attr):
                # Create dropdown for this PSM
                psm_dropdown = QComboBox()
                psm_dropdown.addItems(tool_options)
                # Connect to handler with lambda to pass psm_name
                psm_dropdown.currentTextChanged.connect(
                    lambda text, name=psm_name: self._on_psm_tool_changed(name, text)
                )
                metadata_layout.addRow(f"{psm_name} Tool:", psm_dropdown)
                self.psm_tool_dropdowns[psm_name] = psm_dropdown

                # Create custom input field (hidden by default)
                psm_custom_input = QLineEdit()
                psm_custom_input.setPlaceholderText(f"Enter custom tool name for {psm_name}...")
                psm_custom_input.textChanged.connect(lambda: self._mark_unsaved_changes())
                psm_custom_input.setVisible(False)
                metadata_layout.addRow("", psm_custom_input)
                self.psm_tool_custom_inputs[psm_name] = psm_custom_input

                # Initialize metadata tool section for this PSM if not already present
                if psm_name not in self.metadata['tool']:
                    self.metadata['tool'][psm_name] = ""

    def _toggle_multi_frame_mode(self, state):
        """Toggle multi-frame labeling mode."""
        self.multi_frame_mode = (state == Qt.Checked)

        # Enable/disable range selection buttons
        self.set_start_btn.setEnabled(self.multi_frame_mode)
        self.set_end_btn.setEnabled(self.multi_frame_mode)

        if not self.multi_frame_mode:
            self._clear_frame_range()
            self.statusBar().showMessage("Multi-frame mode disabled")
        else:
            self.statusBar().showMessage("Multi-frame mode enabled - set start and end frames")

    def _set_start_frame(self):
        """Set the start frame for multi-frame labeling."""
        if not self.multi_frame_mode:
            QMessageBox.warning(self, "Multi-Frame Mode Disabled",
                              "Please enable multi-frame mode first.")
            return

        self.frame_range_start = self.current_frame_index
        self._update_range_display()
        self.statusBar().showMessage(f"Start frame set to {self.current_frame_index + 1}")

    def _set_end_frame(self):
        """Set the end frame for multi-frame labeling."""
        if not self.multi_frame_mode:
            QMessageBox.warning(self, "Multi-Frame Mode Disabled",
                              "Please enable multi-frame mode first.")
            return

        if self.frame_range_start is None:
            QMessageBox.warning(self, "Start Frame Not Set",
                              "Please set the start frame first.")
            return

        if self.current_frame_index < self.frame_range_start:
            QMessageBox.warning(self, "Invalid Range",
                              "End frame must be after or equal to start frame. "
                              "The range must be positive (at least 1 frame).")
            return

        self.frame_range_end = self.current_frame_index
        self._update_range_display()
        self.statusBar().showMessage(f"End frame set to {self.current_frame_index + 1}")

    def _clear_frame_range(self):
        """Clear the frame range for multi-frame labeling."""
        self.frame_range_start = None
        self.frame_range_end = None
        self._update_range_display()
        self.statusBar().showMessage("Frame range cleared")

    def _validate_multi_frame_range(self) -> bool:
        """
        Validate that the multi-frame range is valid and positive.

        Returns:
            True if range is valid, False otherwise (with warning message shown)
        """
        if not self.multi_frame_mode:
            return True  # Not in multi-frame mode, so no validation needed

        if self.frame_range_start is None or self.frame_range_end is None:
            QMessageBox.warning(
                self,
                "Invalid Range",
                "Please set both start and end frames for multi-frame labeling."
            )
            return False

        # Check that range is positive (end >= start)
        if self.frame_range_end < self.frame_range_start:
            QMessageBox.warning(
                self,
                "Invalid Range",
                "End frame must be after or equal to start frame. "
                "The range must be positive (at least 1 frame)."
            )
            return False

        return True

    def _update_range_display(self):
        """Update the range display label."""
        if self.frame_range_start is None:
            self.range_display_label.setText("Range: Not set")
        elif self.frame_range_end is None:
            self.range_display_label.setText(f"Range: {self.frame_range_start + 1} - ?")
        else:
            self.range_display_label.setText(
                f"Range: {self.frame_range_start + 1} - {self.frame_range_end + 1} "
                f"({self.frame_range_end - self.frame_range_start + 1} frames)"
            )

    def _add_failure_label(self):
        """Add failure frame range using multi-frame labeling."""
        if not self.frame_files:
            QMessageBox.warning(self, "No Data", "Please load configuration first.")
            return

        # Check if multi-frame mode is enabled and range is set
        if not self.multi_frame_mode:
            QMessageBox.warning(
                self,
                "Multi-Frame Mode Disabled",
                "Please enable multi-frame mode and set start/end frames first."
            )
            return

        # Validate multi-frame range (ensures it's positive and properly set)
        if not self._validate_multi_frame_range():
            return  # Validation failed, warning already shown

        # Add tuple to failure list
        failure_range = (self.frame_range_start, self.frame_range_end)
        self.metadata['failure'].append(failure_range)

        self._mark_unsaved_changes()
        self.statusBar().showMessage(
            f"Failure label added: frames {self.frame_range_start} - {self.frame_range_end}"
        )
        self._update_annotation_list()
        self._update_statistics()

    def _clear_failure_labels(self):
        """Clear all failure labels."""
        if self.metadata['failure']:
            reply = QMessageBox.question(
                self,
                "Clear Failure Labels",
                f"Remove all {len(self.metadata['failure'])} failure range(s)?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.metadata['failure'] = []
                self._mark_unsaved_changes()
                self.statusBar().showMessage("All failure labels cleared")
                self._update_annotation_list()
                self._update_statistics()
        else:
            self.statusBar().showMessage("No failure labels to clear")

    def _add_recovery_label(self):
        """Add recovery frame range using multi-frame labeling."""
        if not self.frame_files:
            QMessageBox.warning(self, "No Data", "Please load configuration first.")
            return

        # Check if multi-frame mode is enabled and range is set
        if not self.multi_frame_mode:
            QMessageBox.warning(
                self,
                "Multi-Frame Mode Disabled",
                "Please enable multi-frame mode and set start/end frames first."
            )
            return

        # Validate multi-frame range (ensures it's positive and properly set)
        if not self._validate_multi_frame_range():
            return  # Validation failed, warning already shown

        # Add tuple to recovery list
        recovery_range = (self.frame_range_start, self.frame_range_end)
        self.metadata['recovery'].append(recovery_range)

        self._mark_unsaved_changes()
        self.statusBar().showMessage(
            f"Recovery label added: frames {self.frame_range_start} - {self.frame_range_end}"
        )
        self._update_annotation_list()
        self._update_statistics()

    def _clear_recovery_labels(self):
        """Clear all recovery labels."""
        if self.metadata['recovery']:
            reply = QMessageBox.question(
                self,
                "Clear Recovery Labels",
                f"Remove all {len(self.metadata['recovery'])} recovery range(s)?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.metadata['recovery'] = []
                self._mark_unsaved_changes()
                self.statusBar().showMessage("All recovery labels cleared")
                self._update_annotation_list()
                self._update_statistics()
        else:
            self.statusBar().showMessage("No recovery labels to clear")

    def _remove_selected_annotation(self):
        """
        Remove selected annotation from the list.

        Works for both multi-frame mode and single annotation removal.
        Removes the selected failure or recovery label from the metadata.
        """
        current_item = self.annotation_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select an annotation to remove.")
            return

        # Get the selected row
        item_index = self.annotation_list.currentRow()
        item_text = current_item.text()

        # Determine if it's a failure or recovery annotation
        if item_text.startswith("Failure"):
            # Extract the failure number from the text (e.g., "Failure 1: Frames 10 - 20")
            failure_count = len(self.metadata['failure'])
            if item_index < failure_count:
                removed_range = self.metadata['failure'][item_index]
                self.metadata['failure'].pop(item_index)
                self.statusBar().showMessage(
                    f"Failure label removed: frames {removed_range[0]} - {removed_range[1]}"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to remove failure annotation.")
                return
        elif item_text.startswith("Recovery"):
            # Calculate the recovery index (offset by number of failures)
            failure_count = len(self.metadata['failure'])
            recovery_index = item_index - failure_count
            if 0 <= recovery_index < len(self.metadata['recovery']):
                removed_range = self.metadata['recovery'][recovery_index]
                self.metadata['recovery'].pop(recovery_index)
                self.statusBar().showMessage(
                    f"Recovery label removed: frames {removed_range[0]} - {removed_range[1]}"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to remove recovery annotation.")
                return
        else:
            QMessageBox.warning(self, "Unknown Annotation", "Cannot determine annotation type.")
            return

        # Update the display
        self._mark_unsaved_changes()
        self._update_annotation_list()
        self._update_statistics()

    def _clear_all_annotations(self):
        """
        Clear all failure and recovery annotations from the metadata.

        Similar to data_annotate.py's clear_all functionality, but adapted
        for metadata (failure and recovery lists).
        """
        # Count total annotations
        total_failure = len(self.metadata['failure'])
        total_recovery = len(self.metadata['recovery'])
        total_annotations = total_failure + total_recovery

        if total_annotations == 0:
            QMessageBox.information(self, "No Annotations", "No failure or recovery labels found.")
            return

        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Clear All Annotations",
            f"Remove ALL {total_annotations} annotations?\n"
            f"  - {total_failure} failure label(s)\n"
            f"  - {total_recovery} recovery label(s)\n\n"
            "This action cannot be undone!",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Clear all annotations
            self.metadata['failure'] = []
            self.metadata['recovery'] = []

            # Update display
            self._mark_unsaved_changes()
            self._update_annotation_list()
            self._update_statistics()
            self.statusBar().showMessage(f"Cleared all {total_annotations} annotations")

    def _update_annotation_list(self):
        """Update the current annotations display."""
        self.annotation_list.clear()

        # Display failure ranges
        if self.metadata['failure']:
            for i, (start, end) in enumerate(self.metadata['failure']):
                item = QListWidgetItem(f"Failure {i+1}: Frames {start} - {end}")
                item.setBackground(QColor(255, 200, 200))  # Light red
                self.annotation_list.addItem(item)

        # Display recovery ranges
        if self.metadata['recovery']:
            for i, (start, end) in enumerate(self.metadata['recovery']):
                item = QListWidgetItem(f"Recovery {i+1}: Frames {start} - {end}")
                item.setBackground(QColor(200, 255, 200))  # Light green
                self.annotation_list.addItem(item)

    def _update_statistics(self):
        """Update statistics display."""
        if not self.frame_files:
            self.stats_label.setText("No data loaded")
            return

        total_frames = len(self.frame_files)

        stats_text = f"""Total Frames: {total_frames}

Metadata:"""

        if self.metadata['user_id']:
            stats_text += f"\n  User ID: {self.metadata['user_id']}"
        if self.metadata['operator_skill_level']:
            stats_text += f"\n  Skill Level: {self.metadata['operator_skill_level']}"
        if self.metadata['case_type']:
            stats_text += f"\n  Case Type: {self.metadata['case_type']}"
        if self.metadata.get('tool'):
            stats_text += "\n  tool:"
            for psm_name, tool_name in self.metadata['tool'].items():
                stats_text += f"\n    {psm_name}: {tool_name}"

        stats_text += "\n\nLabels:"
        if self.metadata['failure']:
            stats_text += f"\n  Failure: {len(self.metadata['failure'])} range(s)"
            for i, (start, end) in enumerate(self.metadata['failure']):
                stats_text += f"\n    {i+1}. Frames {start}-{end}"
        if self.metadata['recovery']:
            stats_text += f"\n  Recovery: {len(self.metadata['recovery'])} range(s)"
            for i, (start, end) in enumerate(self.metadata['recovery']):
                stats_text += f"\n    {i+1}. Frames {start}-{end}"

        self.stats_label.setText(stats_text)

    def _reset_statistics(self):
        """
        Reset/reload statistics display based on current status.

        Recalculates and updates the statistics display to reflect the current
        state of all metadata (user_id, skill level, data type, tool, failure, recovery).
        Useful when you want to refresh the display after making changes.
        """
        self._update_statistics()
        self.statusBar().showMessage("Statistics reloaded")

    def _reset_all_parameters(self):
        """
        Reset all parameters when loading a new configuration.

        Clears all metadata, UI fields, multi-frame state, and annotations
        to ensure a clean slate when switching between different datasets.
        """
        # Reset metadata to initial state
        self.metadata = {
            'user_id': '',
            'operator_skill_level': '',
            'case_type': '',
            'tool': {},
            'failure': [],
            'recovery': []
        }

        # Reset multi-frame labeling state
        self.frame_range_start = None
        self.frame_range_end = None
        if hasattr(self, 'multi_frame_checkbox'):
            self.multi_frame_checkbox.setChecked(False)
        self._update_range_display()

        # Clear UI input fields
        if hasattr(self, 'user_id_input'):
            self.user_id_input.clear()

        if hasattr(self, 'skill_level_dropdown'):
            self.skill_level_dropdown.setCurrentIndex(0)  # Reset to first option (Expert)

        if hasattr(self, 'skill_level_custom_input'):
            self.skill_level_custom_input.clear()
            self.skill_level_custom_input.setVisible(False)

        if hasattr(self, 'case_type_dropdown'):
            self.case_type_dropdown.setCurrentIndex(0)  # Reset to first option (Clinical)

        if hasattr(self, 'case_type_custom_input'):
            self.case_type_custom_input.clear()
            self.case_type_custom_input.setVisible(False)

        # Clear PSM tool controls safely
        # This removes all PSM tool dropdowns and custom inputs from the layout
        # They will be recreated by _create_psm_tool_controls() after this method
        self._clear_psm_tool_controls()

        # Clear annotation list display
        if hasattr(self, 'annotation_list'):
            self.annotation_list.clear()

        # Reset custom save folder (will use config default)
        self.custom_save_folder = None

        # Reset current frame index
        self.current_frame_index = 0

        # Stop auto-play if running
        if hasattr(self, 'is_playing') and self.is_playing:
            self.toggle_auto_play()

        # Update displays
        self._update_annotation_list()
        self._update_statistics()

        # Clear unsaved changes flag after reset
        self.has_unsaved_changes = False

        self.statusBar().showMessage("All parameters reset for new configuration")

    def load_configuration(self):
        """Load configuration and initialize data paths."""
        try:
            # Check for unsaved changes before loading new configuration
            if self.has_unsaved_changes:
                reply = QMessageBox.question(
                    self,
                    "Unsaved Changes",
                    "You have unsaved changes. Do you want to save before loading new configuration?",
                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
                )

                if reply == QMessageBox.Save:
                    self.save_metadata()
                elif reply == QMessageBox.Cancel:
                    return  # Don't load new configuration

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

            # Reset all parameters when loading new configuration
            self._reset_all_parameters()

            # Get image paths from configuration
            self.image_paths = self.config_loader.get_image_paths()

            # Verify paths exist and get frame files
            existing_cameras = []
            for camera_name, camera_path in self.image_paths.items():
                if camera_path.exists():
                    existing_cameras.append(camera_name)

            if not existing_cameras:
                raise FileNotFoundError("No camera image directories found")

            # Load frame list (use first available camera as reference)
            reference_camera = existing_cameras[0]
            self.frame_files = glob_sorted_frame(self.image_paths[reference_camera])

            if not self.frame_files:
                raise ValueError("No image files found")

            # Initialize timeline
            self.timeline_slider.setMaximum(len(self.frame_files) - 1)
            self.timeline_slider.setEnabled(True)
            self.play_pause_btn.setEnabled(True)
            self.speed_combo.setEnabled(True)
            self.frame_jump_input.setMaximum(len(self.frame_files))
            self.frame_jump_input.setValue(1)

            # Enable camera buttons based on available cameras (EXACT same as data_annotate.py)
            self._enable_camera_buttons(existing_cameras)

            # Start image processor thread
            if self.image_processor:
                self.image_processor.stop()
                self.image_processor.wait()

            size_config = self.config_loader.get_image_size_config()
            gui_config = self.config_loader.get_gui_config()
            self.image_processor = ImageProcessor(self.image_paths, self.frame_files, size_config, gui_config)
            self.image_processor.images_loaded.connect(self.on_images_loaded)
            self.image_processor.loading_error.connect(self.on_loading_error)
            self.image_processor.start()

            # Load first frame
            self.current_frame_index = 0
            self.timeline_slider.setValue(0)
            self.image_processor.set_frame_index(0)

            self.statusBar().showMessage(
                f"Configuration loaded - {len(self.frame_files)} frames, cameras: {', '.join(existing_cameras)}"
            )

            # Initialize statistics display
            self._update_statistics()

            # Create PSM tool controls based on configuration
            self._create_psm_tool_controls()

            # Update save folder display if no custom folder is set
            if not self.custom_save_folder:
                config_save_folder = self.config_loader.get_save_folder()
                folder_str = str(config_save_folder)
                if len(folder_str) > 60:
                    folder_str = "..." + folder_str[-57:]
                self.save_folder_label.setText(f"Save to: {folder_str}")

        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", f"Failed to load configuration:\n{str(e)}")
            print(f"Configuration error: {e}")
            import traceback
            traceback.print_exc()

    def load_metadata_file(self):
        """Load existing metadata from JSON file."""
        try:
            # Select JSON file
            json_file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Meta Data File",
                "",
                "JSON files (*.json)"
            )

            if not json_file:
                return

            # Clear all existing annotations before loading new metadata
            self.metadata['failure'] = []
            self.metadata['recovery'] = []
            self.metadata['tool'] = {}

            # Clear multi-frame labeling state
            self.frame_range_start = None
            self.frame_range_end = None
            self._update_range_display()

            # Clear annotation list display
            self.annotation_list.clear()

            # Load JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                loaded_metadata = json.load(f)

            # Validate required fields (Tools is optional for backward compatibility)
            required_fields = ['user_id', 'operator_skill_level', 'case_type', 'failure', 'recovery']
            for field in required_fields:
                if field not in loaded_metadata:
                    raise ValueError(f"Missing required field: {field}")

            # Ensure tool field exists (for backward compatibility with old files)
            if 'tool' not in loaded_metadata:
                loaded_metadata['tool'] = {}

            # Ensure failure and recovery are lists (for backward compatibility)
            if not isinstance(loaded_metadata['failure'], list):
                # Old format was single integer, convert to empty list
                loaded_metadata['failure'] = []
            if not isinstance(loaded_metadata['recovery'], list):
                # Old format was single integer, convert to empty list
                loaded_metadata['recovery'] = []

            # Convert list items to tuples if needed
            loaded_metadata['failure'] = [tuple(item) if isinstance(item, list) else item
                                          for item in loaded_metadata['failure']]
            loaded_metadata['recovery'] = [tuple(item) if isinstance(item, list) else item
                                           for item in loaded_metadata['recovery']]

            # Update metadata
            self.metadata = loaded_metadata

            # Update UI fields
            self.user_id_input.setText(str(self.metadata['user_id']))

            # Skill level
            skill_level = self.metadata['operator_skill_level']
            if skill_level in ["Expert", "Intermediate", "Novice"]:
                self.skill_level_dropdown.setCurrentText(skill_level)
            else:
                self.skill_level_dropdown.setCurrentText("Others")
                self.skill_level_custom_input.setText(skill_level)

            # Case type
            case_type = self.metadata['case_type']
            if case_type in ["Clinical", "Ex-vivo", "Table-Top Phantom", "Digital Simulation", "Physical Simulation"]:
                self.case_type_dropdown.setCurrentText(case_type)
            else:
                self.case_type_dropdown.setCurrentText("Others")
                self.case_type_custom_input.setText(case_type)

            # PSM tool - populate dropdowns if they exist
            if 'tool' in self.metadata and self.metadata['tool']:
                for psm_name, tool_name in self.metadata['tool'].items():
                    if psm_name in self.psm_tool_dropdowns:
                        dropdown = self.psm_tool_dropdowns[psm_name]
                        # Check if tool_name is one of the predefined options
                        tool_options = ["Large_Needle_Driver", "Prograsp_Forceps",
                                      "Maryland_Bipolar_Forceps", "Curved_Scissors"]
                        if tool_name in tool_options:
                            dropdown.setCurrentText(tool_name)
                        else:
                            # Custom tool name
                            dropdown.setCurrentText("Others")
                            if psm_name in self.psm_tool_custom_inputs:
                                self.psm_tool_custom_inputs[psm_name].setText(tool_name)

            # Update displays
            self._update_annotation_list()
            self._update_statistics()

            # Clear unsaved changes flag after loading
            self.has_unsaved_changes = False

            self.statusBar().showMessage(f"Metadata loaded from {Path(json_file).name}")

            QMessageBox.information(
                self,
                "Metadata Loaded",
                f"Successfully loaded metadata from:\n{json_file}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load metadata:\n{str(e)}")
            print(f"Load error: {e}")
            import traceback
            traceback.print_exc()

    def select_save_folder(self):
        """Select custom save folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Save Folder",
            str(Path.home())
        )

        if folder:
            self.custom_save_folder = Path(folder)
            folder_str = str(self.custom_save_folder)
            if len(folder_str) > 60:
                folder_str = "..." + folder_str[-57:]
            self.save_folder_label.setText(f"Save to: {folder_str}")
            self.statusBar().showMessage(f"Save folder set to: {self.custom_save_folder}")

    def save_metadata(self):
        """Save metadata to meta_data.json file."""
        try:
            # Get current metadata from UI
            self.metadata['user_id'] = self.user_id_input.text().strip()

            # Skill level
            if self.skill_level_dropdown.currentText() == "Others":
                self.metadata['operator_skill_level'] = self.skill_level_custom_input.text().strip()
            else:
                self.metadata['operator_skill_level'] = self.skill_level_dropdown.currentText()

            # Case type
            if self.case_type_dropdown.currentText() == "Others":
                self.metadata['case_type'] = self.case_type_custom_input.text().strip()
            else:
                self.metadata['case_type'] = self.case_type_dropdown.currentText()

            # PSM tool - collect from dropdowns
            self.metadata['tool'] = {}
            for psm_name, dropdown in self.psm_tool_dropdowns.items():
                if dropdown.currentText() == "Others":
                    # Use custom input if "Others" is selected
                    if psm_name in self.psm_tool_custom_inputs:
                        tool_name = self.psm_tool_custom_inputs[psm_name].text().strip()
                        if tool_name:  # Only add if non-empty
                            self.metadata['tool'][psm_name] = tool_name
                else:
                    # Use dropdown selection
                    self.metadata['tool'][psm_name] = dropdown.currentText()

            # Get save folder
            if self.custom_save_folder:
                save_folder = self.custom_save_folder
            elif self.config:
                save_folder = self.config_loader.get_save_folder()
            else:
                QMessageBox.warning(self, "No Save Folder", "Please select a save folder first.")
                return

            # Create folder if it doesn't exist
            if not save_folder.exists():
                create_folder(save_folder)

            # Save to meta_data.json
            meta_data_file = save_folder / "meta_data.json"
            with open(meta_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

            self.statusBar().showMessage(f"Metadata saved to {meta_data_file}")

            # Clear unsaved changes flag after saving
            self.has_unsaved_changes = False

            # Build summary message
            failure_summary = "Not set"
            if self.metadata['failure']:
                failure_summary = f"{len(self.metadata['failure'])} range(s): " + ", ".join(
                    [f"({start}-{end})" for start, end in self.metadata['failure']]
                )

            recovery_summary = "Not set"
            if self.metadata['recovery']:
                recovery_summary = f"{len(self.metadata['recovery'])} range(s): " + ", ".join(
                    [f"({start}-{end})" for start, end in self.metadata['recovery']]
                )

            # tool summary
            tool_summary = "Not set"
            if self.metadata['tool']:
                tool_summary = ", ".join([f"{psm}: {tool}" for psm, tool in self.metadata['tool'].items()])

            QMessageBox.information(
                self,
                "Save Complete",
                f"Metadata successfully saved to:\n{meta_data_file}\n\n"
                f"User ID: {self.metadata['user_id']}\n"
                f"Skill Level: {self.metadata['operator_skill_level']}\n"
                f"Case Type: {self.metadata['case_type']}\n"
                f"tool: {tool_summary}\n"
                f"Failure: {failure_summary}\n"
                f"Recovery: {recovery_summary}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save metadata:\n{str(e)}")
            print(f"Save error: {e}")
            import traceback
            traceback.print_exc()

    def get_save_folder(self) -> Path:
        """Get the current save folder (custom or from config)."""
        if self.custom_save_folder:
            return self.custom_save_folder
        elif self.config:
            return self.config_loader.get_save_folder()
        else:
            return Path.home() / "dvrk_metadata"

    def _on_timeline_changed(self, value: int):
        """Handle timeline slider changes (EXACT same as data_annotate.py)."""
        if self.image_processor:
            self.current_frame_index = value
            self.image_processor.set_frame_index(value)
            self.frame_jump_input.setValue(value + 1)

    def jump_to_frame(self, frame_index: int):
        """Jump to specific frame (EXACT same as data_annotate.py)."""
        if 0 <= frame_index < len(self.frame_files):
            self.timeline_slider.setValue(frame_index)

    def previous_frame(self):
        """Go to previous frame (EXACT same as data_annotate.py)."""
        if self.current_frame_index > 0:
            self.jump_to_frame(self.current_frame_index - 1)

    def next_frame(self):
        """Go to next frame (EXACT same as data_annotate.py)."""
        if self.current_frame_index < len(self.frame_files) - 1:
            self.jump_to_frame(self.current_frame_index + 1)

    def _open_original_image(self, camera_name: str):
        """
        Open the original (unresized) image for the specified camera in a new window (EXACT same as data_annotate.py).

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
        Create a separate, independent window to display the original image (EXACT same as data_annotate.py).

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

    def toggle_auto_play(self):
        """
        Toggle auto-play functionality with configurable speed (EXACT same as data_annotate.py).

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
        """Advance to next frame during auto-play (EXACT same as data_annotate.py)."""
        if self.current_frame_index < len(self.frame_files) - 1:
            self.next_frame()
        else:
            # Reached end, stop auto-play
            self.toggle_auto_play()
            self.statusBar().showMessage("Auto-play completed")

    def _on_speed_combo_changed(self, speed_text: str):
        """
        Handle speed combo box changes with real-time status display (EXACT same as data_annotate.py).

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
                start_frame = self.current_frame_index + 1  # 1-based for display
                self.playback_status_label.setText(f"{frequency_hz:.1f}Hz | Start: {start_frame}")
                self.statusBar().showMessage(f"Auto-play: {speed_text} ({frequency_hz:.1f} Hz)")

                # Update timer if currently playing
                self.auto_play_timer.setInterval(speed_ms)
            else:
                self.playback_status_label.setText(f"Ready - {speed_text}")
                self.statusBar().showMessage(f"Speed set to {speed_text} ({frequency_hz:.1f} Hz)")

    def _enable_camera_buttons(self, existing_cameras: List[str]):
        """
        Enable camera buttons based on available cameras (EXACT same as data_annotate.py).

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

    def on_images_loaded(self, combined_image, frame_info):
        """Handle loaded images from background thread."""
        # Convert numpy array to QPixmap for display
        height, width, channel = combined_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(combined_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)

        # Display on label
        self.video_display_label.setPixmap(pixmap)

        # Update frame info
        self.frame_info_label.setText(frame_info)

    def on_loading_error(self, error_message):
        """Handle image loading errors."""
        self.statusBar().showMessage(f"Error: {error_message}")

    def closeEvent(self, event):
        """Handle window close event with unsaved changes warning."""
        # Check for unsaved changes
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )

            if reply == QMessageBox.Save:
                self.save_metadata()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return

        # Stop image processor thread
        if self.image_processor:
            self.image_processor.stop()
            self.image_processor.wait()

        # Stop auto-play timer
        if self.auto_play_timer:
            self.auto_play_timer.stop()

        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = MetaDataAnnotationGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
