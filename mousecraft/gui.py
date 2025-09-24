import sys
import cv2
import json
import pandas as pd
import os
from PyQt5.QtWidgets import QMessageBox, QFileDialog
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QFileDialog, QSpinBox, QDoubleSpinBox, 
    QGroupBox, QGridLayout, QTextEdit, QComboBox, QCheckBox,
    QMessageBox, QProgressBar, QSplitter, QLineEdit, QSizePolicy,
    QDialog, QFrame, QSpacerItem, QColorDialog
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QIntValidator, QIcon, QFont, QFontDatabase, QMovie
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QSettings
from PyQt5.QtCore import QProcess
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from PyQt5.QtWidgets import QStyle
import csv
from PyQt5.QtWidgets import QScrollArea
from collections import Counter
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class DraggableTimeline(FigureCanvas):
    """Custom matplotlib canvas with draggable timeline"""
    timeline_moved = pyqtSignal(int)
    timeline_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4.0))  # Increased height for better visibility
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        # Ensure margins so x ticks aren't hidden and top is neat
        try:
            self.fig.subplots_adjust(bottom=0.25, top=0.95)
        except Exception:
            pass
        self.current_frame = 0
        self.total_frames = 0
        self.motion_energy = None
        self.onsets = []
        self.onset_types = {}  # 'active' or 'twitch'
        self.onset_validations = {}  # 'accepted', 'rejected', 'edited'
        self.event_offsets = {}  # Store offset frames for events
        
        self.timeline_line = None
        self.dragging = False
        self.connect_events()
        
    def connect_events(self):
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        
    def on_press(self, event):
        # Notify listeners that this timeline has been interacted with
        try:
            self.timeline_clicked.emit()
        except Exception:
            pass
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Left click
            self.dragging = True
            self.current_frame = int(event.xdata)
            self.update_timeline()
            self.timeline_moved.emit(self.current_frame)
            
    def on_release(self, event):
        self.dragging = False
        
    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        if event.xdata is not None:
            self.current_frame = max(0, min(self.total_frames - 1, int(event.xdata)))
            self.update_timeline()
            self.timeline_moved.emit(self.current_frame)
            
    def update_timeline(self):
        try:
            if self.timeline_line is not None:
                self.timeline_line.remove()
        except Exception:
            pass
        self.timeline_line = self.ax.axvline(self.current_frame, color='red', linewidth=2, alpha=0.8)
        self.draw()
        
        
    def plot_motion_energy(self, motion_energy, onsets=None, onset_types=None, event_offsets=None, event_status=None):
        import numpy as np

        self.motion_energy = motion_energy
        self.total_frames = len(motion_energy)
        self.onsets = onsets or []
        self.onset_types = onset_types or {}
        self.event_offsets = event_offsets or {}
        self.event_status = event_status or {}
        # Ensure colors map exists
        if not hasattr(self, 'event_type_colors') or self.event_type_colors is None:
            self.event_type_colors = {
                'twitch': 'purple',
                'active': 'yellow',
                'complex': 'cyan',
            }

        self.ax.clear()
        self.ax.plot(np.arange(self.total_frames), self.motion_energy, color='#1f77b4', linewidth=1)

        # Determine which event types to display (if restricted)
        allowed_types = getattr(self, 'visible_event_types', None)
        # Plot onset-to-offset spans
        for onset in self.onsets:
            onset_type = self.onset_types.get(onset, '')
            if allowed_types is not None and str(onset_type).lower() not in {t.lower() for t in allowed_types}:
                continue
            offset = self.event_offsets.get(onset, onset)
            status = self.event_status.get(onset, '')
            alpha = 1.0 if status == 'accepted' else 0.4
            # Pick color from mapping with graceful fallback
            color_key = str(onset_type).lower()
            color_value = self.event_type_colors.get(color_key, self.event_type_colors.get(onset_type, None))
            if color_value is None:
                # Default colors: twitch purple, active yellow, complex cyan
                if color_key == 'twitch':
                    color_value = 'purple'
                elif color_key == 'active':
                    color_value = 'yellow'
                elif color_key == 'complex':
                    color_value = 'cyan'
                else:
                    color_value = '#888888'
            self.ax.axvspan(onset, offset, color=color_value, alpha=alpha)

        # Add validation markers as bullet points on top
        for onset in self.onsets:
            validation = self.onset_validations.get(onset, 'pending')
            # Skip pending events (no marker)
            if validation == 'pending':
                continue
            # Compute score for color logic
            score = 0
            if validation == 'edited':
                # Check if this edited event has a score in event_status
                if hasattr(self, 'event_status') and onset in self.event_status:
                    score = self.event_status[onset]
                else:
                    score = 0.5  # Default for edited events
            elif validation == 'accepted':
                score = 1
            elif validation == 'rejected':
                score = -1
            elif validation == 'manually added':
                score = 0
            else:
                score = 0
            # Assign color based on score
            if score == 1:
                color = 'green'
            elif score == 0.5:
                color = 'orange'
            elif score == -1:
                color = 'red'
            elif validation == 'manually added':
                color = '#00CED1'  # Turquoise blue for manually added events
            else:
                color = 'gray'  # Fallback for any other cases
            self.ax.plot(onset, max(self.motion_energy) * 1.05, 'o', color=color, markersize=6)

        self.ax.set_xlim(0, max(self.total_frames, 1000))
        self.ax.set_ylim(0, max(self.motion_energy) * 1.2)  # Increased ylim to accommodate bullets
        self.ax.set_ylabel("motion_energy", fontsize=7)
        self.ax.set_xlabel("Frame", fontsize=7)
        # Remove top/right spines and ensure only bottom/left axes are shown
        try:
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.xaxis.set_ticks_position('bottom')
            self.ax.yaxis.set_ticks_position('left')
        except Exception:
            pass
        self.ax.set_title("classified motion", fontsize=9)
        self.ax.set_yticks([])
        self.ax.tick_params(axis='x', labelsize=6)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.update_timeline()
        
    def plot_motion_energy_preserve_view(self, *args, **kwargs):
        try:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
        except Exception:
            xlim = None
            ylim = None
        self.plot_motion_energy(*args, **kwargs)
        if xlim is not None and ylim is not None:
            try:
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
            except Exception:
                pass
        self.update_timeline()
        self.draw()

    def set_onset_validation(self, onset, validation):
        # Store current view limits
        # xlim = self.ax.get_xlim()
        # ylim = self.ax.get_ylim()
        self.onset_validations[onset] = validation
        self.plot_motion_energy_preserve_view(self.motion_energy, self.onsets, self.onset_types, self.event_offsets, self.event_status)
        # Restore view limits
        # self.ax.set_xlim(xlim)
        # self.ax.set_ylim(ylim)
        self.draw()
        
    def add_event(self, onset, event_type, offset=None):
        """Add a new event to the timeline"""
        self.onsets.append(onset)
        self.onset_types[onset] = event_type
        if offset is not None:
            self.event_offsets[onset] = offset
        self.plot_motion_energy_preserve_view(self.motion_energy, self.onsets, self.onset_types, self.event_offsets, self.event_status)
        
    def remove_event(self, onset):
        """Remove an event from the timeline"""
        if onset in self.onsets:
            self.onsets.remove(onset)
        if onset in self.onset_types:
            del self.onset_types[onset]
        if onset in self.onset_validations:
            del self.onset_validations[onset]
        if onset in self.event_offsets:
            del self.event_offsets[onset]
        self.plot_motion_energy_preserve_view(self.motion_energy, self.onsets, self.onset_types, self.event_offsets, self.event_status)

    def wheelEvent(self, event):
        # Zoom in/out on scroll, centered on current_frame
        ax = self.ax
        xlim = ax.get_xlim()
        center = self.current_frame
        width = xlim[1] - xlim[0]
        if event.angleDelta().y() > 0:
            # Zoom in
            new_width = width / 2
        else:
            # Zoom out
            new_width = width * 2
        new_xlim = (max(center - new_width/2, 0), min(center + new_width/2, self.total_frames))
        ax.set_xlim(new_xlim)
        self.draw()
        event.accept()

class FrameSlider(QSlider):
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            event.ignore()
        else:
            super().keyPressEvent(event)

class MotionAnnotator(QWidget):
    """J'ai rajouté cette fonction"""
    def find_closest_onset_idx(self, frame):
        """Return the index of the closest onset <= frame, or 0 if none."""
        if not self.onsets:
            return 0
        idx = 0
        for i, onset in enumerate(self.onsets):
            if onset <= frame:
                idx = i
            else:
                break
        return idx

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MouseCraft")
        self.setGeometry(100, 100, 1400, 900)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.start_mouse_timer()
        # Set window icon to mouse.webp if possible, else fallback to MouseCraft.png
        mouse_icon_path = os.path.join(SCRIPT_DIR, 'resources',"mouse.png")
        mousecraft_icon_path = os.path.join(SCRIPT_DIR, 'resources', "MouseCraft.png")
        if os.path.exists(mouse_icon_path):
            try:
                self.setWindowIcon(QIcon(mouse_icon_path))
            except Exception:
                if os.path.exists(mousecraft_icon_path):
                    self.setWindowIcon(QIcon(mousecraft_icon_path))
        elif os.path.exists(mousecraft_icon_path):
            self.setWindowIcon(QIcon(mousecraft_icon_path))
        
        # Data storage
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 1  # Initialisation par défaut à 1 dans __init__
        self.current_frame = 0
        self.playback_speed = 1.0
        
        self.motion_energy = None
        self.classified_events = {}
        self.curated_events = {}
        self.onsets = []
        self.onset_types = {}  # 'active' or 'twitch'
        self.current_onset_idx = 0
        
        # Performance tracking
        self.performance_metrics = {
            'true_positives': 0,
            'false_positives': 0,
        }
        
        self.awaiting_offset_validation = False
        self.current_offset_for_validation = None
        self.undo_stack = []  # Pile d'historique pour undo

        # 1. In __init__ of MotionAnnotator, add storage for original onsets
        self.original_onsets = {}  # Maps current onset to original input onset
        # 1. In __init__, add storage for original offsets
        self.original_offsets = {}  # Maps current onset to original input offset
        
        # Threshold for edited events score (default: 5 frames)
        self.edit_threshold = 5

        # In __init__ (add these attributes):
        self._mouse_half_shown = False
        self._mouse_milestones = set()
        self._mouse_milestones_shown = set()

        # Dynamic event types and colors
        self.available_event_types = ["twitch", "active", "complex"]
        self.event_type_colors = {
            "twitch": "purple",
            "active": "yellow",
            "complex": "cyan",
        }

        # Palette de couleurs pour nouveaux types (cycle stable)
        self._auto_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        self.init_ui()
        self.setup_timer()
        self.unsaved_changes = False  # Track unsaved changes
        self.setup_auto_save(interval_minutes=20)

        
    def init_ui(self):
        main_layout = QVBoxLayout()
        # Add MouseCraft logo at the top, centered
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = os.path.join(SCRIPT_DIR, 'resources', "MouseCraft.png")
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_label.setPixmap(logo_pixmap.scaledToWidth(300, Qt.SmoothTransformation))
        else:
            logo_label.setText("<b>MouseCraft</b>")
            logo_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #333;")
        main_layout.addWidget(logo_label)
        content_layout = QHBoxLayout()

        # Left panel - Video and controls
        left_panel = QVBoxLayout()

        # Add vertical spacer between logo and controls
        left_panel.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Video folder label (above video)
        self.video_folder_label = QLabel("")
        self.video_folder_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.video_folder_label)

        # Video display area: stack for up to 2 cameras, wrapped in a fixed-height container when splitting
        # Independent zoom factors per camera (do not affect layout size)
        self.video_zoom_factor_cam1 = 1.0
        self.video_zoom_factor_cam2 = 1.0
        self.video_container = QWidget()
        self.video_stack_layout = QVBoxLayout()
        self.video_stack_layout.setContentsMargins(0, 0, 0, 0)
        self.video_stack_layout.setSpacing(2)
        # Camera 1
        self.video_label = QLabel("Load a video to start")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(600, 450)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        self.video_label.installEventFilter(self)
        self.video_stack_layout.addWidget(self.video_label)
        # Camera 2 (initially hidden)
        self.second_video_label = QLabel("")
        self.second_video_label.setAlignment(Qt.AlignCenter)
        self.second_video_label.setMinimumSize(0, 0)
        self.second_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.second_video_label.setStyleSheet("border: 2px solid gray;")
        self.second_video_label.installEventFilter(self)
        self.second_video_label.hide()
        self.video_stack_layout.addWidget(self.second_video_label)
        # Default: single camera uses full space; second is collapsed
        self.video_stack_layout.setStretch(0, 1)
        self.video_stack_layout.setStretch(1, 0)
        self.video_container.setLayout(self.video_stack_layout)
        self.video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_container.setMinimumHeight(450)
        left_panel.addWidget(self.video_container, 1)

        # Video controls
        video_controls = QGroupBox("Video Controls")
        video_layout = QGridLayout()
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        video_layout.addWidget(self.load_video_btn, 0, 0)
        # Add a camera button
        self.add_camera_btn = QPushButton("Add a camera")
        self.add_camera_btn.clicked.connect(self.add_camera)
        video_layout.addWidget(self.add_camera_btn, 0, 1)
        
        # Compute motion energy by running the analysis script
        self.compute_motion_energy_btn = QPushButton("Compute a ME")
        self.compute_motion_energy_btn.clicked.connect(self.run_state_detection_script)
        video_layout.addWidget(self.compute_motion_energy_btn, 0, 2)
        # Initially disabled until a motion energy file is selected/loaded
        self.compute_motion_energy_btn.setEnabled(False)

        # Rendre les trois boutons de même taille
        self.load_video_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.add_camera_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.compute_motion_energy_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Création des boutons de contrôle vidéo
        self.play_btn = QPushButton("Play ▶️")
        self.loop_btn = QPushButton("Loop 🔁")
        self.pause_btn = QPushButton("Pause ⏸")
        self.stop_btn = QPushButton("Stop ⏹")
        # Forcer activation/visibilité
        self.play_btn.setEnabled(True)
        self.loop_btn.setEnabled(False)  # Enabled only when current frame is an onset
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.play_btn.setVisible(True)
        self.loop_btn.setVisible(True)
        self.pause_btn.setVisible(True)
        self.stop_btn.setVisible(True)
        self.play_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.loop_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.pause_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.stop_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.play_btn.setMinimumHeight(34)
        self.loop_btn.setMinimumHeight(34)
        self.pause_btn.setMinimumHeight(34)
        self.stop_btn.setMinimumHeight(34)
        # Diagnostic widgets recouvrants
        for child in self.findChildren(QWidget):
            if child is not self.pause_btn and child.geometry().intersects(self.pause_btn.geometry()):
                print("Widget potentiellement recouvrant :", child, "visible ?", child.isVisible(), "geometry :", child.geometry())

        # Connexions
        self.play_btn.clicked.connect(self.play)
        self.loop_btn.clicked.connect(self.start_loop_playback)
        self.pause_btn.clicked.connect(lambda: self.pause_btn.setFocus())
        self.pause_btn.pressed.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)

        # Layout horizontal pour les boutons
        play_pause_layout = QHBoxLayout()
        play_pause_layout.addWidget(self.play_btn)
        play_pause_layout.addWidget(self.loop_btn)
        play_pause_layout.addWidget(self.pause_btn)
        self.pause_btn.raise_()  # monte le bouton tout en haut
        play_pause_layout.addWidget(self.stop_btn)
        video_layout.addLayout(play_pause_layout, 1, 0, 1, 4)

        # Initialize loop button enabled state based on initial frame
        self.loop_btn.setEnabled(False)

        # FPS controls on third row
        self.fps_lineedit = QLineEdit()
        self.fps_lineedit.setPlaceholderText("FPS")
        self.fps_lineedit.setValidator(QIntValidator(1, 1000, self))
        self.fps_lineedit.setText('1')
        self.fps_lineedit.textChanged.connect(self.handle_fps_change)
        # self.fps_lineedit.setStyleSheet("background: rgba(255,255,0, 0.5);") # Supprimé
        video_layout.addWidget(QLabel("FPS:"), 2, 0)
        video_layout.addWidget(self.fps_lineedit, 2, 1)

        # Ajout du layout au groupbox
        video_controls.setLayout(video_layout)
        left_panel.addWidget(video_controls)

        # Frame slider
        self.frame_slider = FrameSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.slider_moved)
        left_panel.addWidget(self.frame_slider)

        # Frame info (replace label with spinbox + label)
        frame_info_layout = QHBoxLayout()
        frame_info_layout.setContentsMargins(0, 0, 0, 0)
        frame_info_layout.setSpacing(4)
        frame_info_layout.addWidget(QLabel("Frame:"))
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setRange(0, self.total_frames - 1)
        self.frame_spinbox.setValue(self.current_frame)
        self.frame_spinbox.setFixedWidth(70)
        self.frame_spinbox.valueChanged.connect(self.frame_spinbox_changed)
        frame_info_layout.addWidget(self.frame_spinbox)
        frame_info_layout.addWidget(QLabel("/"))
        self.total_frames_label = QLabel(str(self.total_frames))
        frame_info_layout.addWidget(self.total_frames_label)
        frame_info_layout.addStretch(1)
        left_panel.addLayout(frame_info_layout)
        # Remove: self.frame_info_label = QLabel("Frame: 0 / 0")
        # left_panel.addWidget(self.frame_info_label)

        # Add dual onset status bar below video (left = input 1, right = input 2)
        status_container = QWidget()
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(0)
        self.onset_status_label_left = QLabel("Frame 0: No onset")
        self.onset_status_label_left.setAlignment(Qt.AlignCenter)
        self.onset_status_label_left.setStyleSheet("background-color: lightgray; padding: 5px; border: 1px solid gray;")
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setLineWidth(1)
        self.onset_status_label_right = QLabel("Frame 0: No onset")
        self.onset_status_label_right.setAlignment(Qt.AlignCenter)
        self.onset_status_label_right.setStyleSheet("background-color: lightgray; padding: 5px; border: 1px solid gray;")
        status_layout.addWidget(self.onset_status_label_left)
        status_layout.addWidget(separator)
        status_layout.addWidget(self.onset_status_label_right)
        status_layout.setStretch(0, 1)
        status_layout.setStretch(2, 1)
        status_container.setLayout(status_layout)
        left_panel.addWidget(status_container)

        # Manual event addition
        manual_event_group = QGroupBox("Manual Event Addition")
        manual_layout = QVBoxLayout()
        event_type_layout = QHBoxLayout()
        event_type_layout.addWidget(QLabel("Event Type:"))
        self.event_type_combo = QComboBox()
        self.event_type_combo.addItem("Select type…")  # Placeholder
        self.event_type_combo.addItems(self.available_event_types)
        self.event_type_combo.setCurrentIndex(0)
        event_type_layout.addWidget(self.event_type_combo)
        # Add a button to add new event types
        self.add_type_btn = QPushButton("Add a type")
        self.add_type_btn.clicked.connect(self.open_add_type_dialog)
        event_type_layout.addWidget(self.add_type_btn)
        manual_layout.addLayout(event_type_layout)
        onset_offset_layout = QGridLayout()
        self.onset_spinbox = QSpinBox()
        self.onset_spinbox.setRange(0, 999999)
        self.onset_spinbox.valueChanged.connect(self.update_offset_range)
        self.offset_spinbox = QSpinBox()
        self.offset_spinbox.setRange(0, 999999)
        self.set_current_onset_btn = QPushButton("Set Current Frame as Onset")
        self.set_current_onset_btn.clicked.connect(self.set_current_as_onset)
        self.set_current_offset_btn = QPushButton("Set Current Frame as Offset")
        self.set_current_offset_btn.clicked.connect(self.set_current_as_offset)
        onset_offset_layout.addWidget(QLabel("Onset Frame:"), 0, 0)
        onset_offset_layout.addWidget(self.onset_spinbox, 0, 1)
        onset_offset_layout.addWidget(self.set_current_onset_btn, 0, 2)
        onset_offset_layout.addWidget(QLabel("Offset Frame:"), 1, 0)
        onset_offset_layout.addWidget(self.offset_spinbox, 1, 1)
        onset_offset_layout.addWidget(self.set_current_offset_btn, 1, 2)
        manual_layout.addLayout(onset_offset_layout)
        
        self.add_event_btn = QPushButton("➕ Add Event")
        self.add_event_btn.clicked.connect(self.add_manual_event)
        manual_layout.addWidget(self.add_event_btn)
        
        # Add threshold control for edited events
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Edit Threshold (frames):"))
        self.edit_threshold_spinbox = QSpinBox()
        self.edit_threshold_spinbox.setRange(1, 100)
        self.edit_threshold_spinbox.setValue(self.edit_threshold)
        self.edit_threshold_spinbox.valueChanged.connect(self.update_edit_threshold)
        threshold_layout.addWidget(self.edit_threshold_spinbox)
        manual_layout.addLayout(threshold_layout)
        manual_event_group.setLayout(manual_layout)
        left_panel.addWidget(manual_event_group)

        # Right panel - Motion energy and annotation
        right_panel = QVBoxLayout()
        # Motion energy grandparent folder label (above timeline)
        self.motion_energy_folder_label = QLabel("")
        self.motion_energy_folder_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.motion_energy_folder_label)
        timeline_group = QGroupBox("Motion Energy Timeline")
        timeline_layout = QVBoxLayout()
        # Container to constrain total timeline height when using two stacked timelines
        timeline_container = QWidget()
        timeline_container_layout = QVBoxLayout()
        timeline_container_layout.setContentsMargins(0, 0, 0, 0)
        timeline_container_layout.setSpacing(2)
        self.timeline_splitter = QSplitter(Qt.Vertical)
        # First timeline (primary)
        self.timeline_canvas = DraggableTimeline()
        # Sync event type colors mapping to timeline canvas
        try:
            self.timeline_canvas.event_type_colors = dict(self.event_type_colors)
        except Exception:
            pass
        # By default, hide complex events (match your first-case display)
        self.timeline_canvas.visible_event_types = ['twitch', 'active'] # add complex if you want to see it displayed 
        self.timeline_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.timeline_canvas.setMinimumHeight(200)
        self.timeline_canvas.setMaximumHeight(500)
        self.timeline_canvas.timeline_moved.connect(self.timeline_frame_changed_primary)
        
        def _focus_primary():
            self.active_timeline_index = 1
            if hasattr(self, 'annotate_primary_radio'):
                self.annotate_primary_radio.setChecked(True)
        self.timeline_canvas.timeline_clicked.connect(_focus_primary)
        # Second timeline (secondary input)
        self.timeline_canvas2 = DraggableTimeline()
        try:
            self.timeline_canvas2.event_type_colors = dict(self.event_type_colors)
        except Exception:
            pass
        # Keep same visibility policy by default
        self.timeline_canvas2.visible_event_types = ['twitch', 'active']
        self.timeline_canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.timeline_canvas2.setMinimumHeight(200)
        self.timeline_canvas2.setMaximumHeight(500)
        self.timeline_canvas2.timeline_moved.connect(self.timeline_frame_changed_secondary)
        def _focus_secondary():
            self.active_timeline_index = 2
            if hasattr(self, 'annotate_secondary_radio'):
                self.annotate_secondary_radio.setChecked(True)
        self.timeline_canvas2.timeline_clicked.connect(_focus_secondary)
        # Add both to vertical splitter; hide second until used
        self.timeline_splitter.addWidget(self.timeline_canvas)
        self.timeline_splitter.addWidget(self.timeline_canvas2)
        self.timeline_canvas2.hide()
        # Equal space when both visible
        self.timeline_splitter.setSizes([1, 1])
        timeline_container_layout.addWidget(self.timeline_splitter)
        timeline_container.setLayout(timeline_container_layout)
        # Constrain the combined timelines height so controls below stay visible
        timeline_container.setMaximumHeight(520)
        timeline_container.setMinimumHeight(400)
        timeline_layout.addWidget(timeline_container)
        # Secondary signal annotations store (separate from primary)
        self.onsets2 = []
        self.onset_types2 = {}
        self.active_timeline_index = 1
        
        # Add custom zoom in/out and reset buttons (same style as video zoom)
        zoom_btn_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton('🔍+')
        self.zoom_out_btn = QPushButton('🔍-')
        self.reset_zoom_btn = QPushButton('Reset Zoom')
        self.zoom_in_btn.clicked.connect(self.zoom_in_timeline)
        self.zoom_out_btn.clicked.connect(self.zoom_out_timeline)
        self.reset_zoom_btn.clicked.connect(self.reset_zoom_timeline)
        zoom_btn_layout.addWidget(self.zoom_in_btn)
        zoom_btn_layout.addWidget(self.zoom_out_btn)
        zoom_btn_layout.addWidget(self.reset_zoom_btn)
        timeline_layout.addLayout(zoom_btn_layout)
        
        timeline_controls = QHBoxLayout()
        self.load_input_btn = QPushButton("Load an input")
        self.load_input_btn.clicked.connect(self.load_input)
        self.add_second_input_btn = QPushButton("Add a second input")
        self.add_second_input_btn.clicked.connect(self.load_second_input)
        timeline_controls.addWidget(self.load_input_btn)
        timeline_controls.addWidget(self.add_second_input_btn)
        # Add explicit annotation target selectors
        annotate_target_layout = QHBoxLayout()
        from PyQt5.QtWidgets import QRadioButton
        self.annotate_primary_radio = QRadioButton("Annotate input 1")
        self.annotate_secondary_radio = QRadioButton("Annotate input 2")
        self.annotate_primary_radio.setChecked(True)
        self.annotate_secondary_radio.setEnabled(False)
        def _set_active_primary():
            if self.annotate_primary_radio.isChecked():
                self.active_timeline_index = 1
        def _set_active_secondary():
            if self.annotate_secondary_radio.isChecked():
                self.active_timeline_index = 2
        self.annotate_primary_radio.toggled.connect(_set_active_primary)
        self.annotate_secondary_radio.toggled.connect(_set_active_secondary)
        annotate_target_layout.addWidget(self.annotate_primary_radio)
        annotate_target_layout.addWidget(self.annotate_secondary_radio)
        annotate_target_layout.addStretch(1)
        timeline_layout.addLayout(timeline_controls)
        timeline_layout.addLayout(annotate_target_layout)
        timeline_group.setLayout(timeline_layout)
        onset_group = QGroupBox("Onset Navigation & Validation")
        onset_layout = QVBoxLayout()
        # Add filter combo boxes side by side
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Event type"))
        self.onset_filter_combo = QComboBox()
        self.onset_filter_combo.addItems(["All"] + [t.capitalize() for t in self.available_event_types])
        self.onset_filter_combo.currentIndexChanged.connect(self.update_onset_filter)
        filter_layout.addWidget(self.onset_filter_combo)
        filter_layout.addWidget(QLabel("Event status"))
        self.status_filter_combo = QComboBox()
        self.status_filter_combo.addItems(["All", "Edited", "Pending", "Accepted", "Rejected", "Manually Added"])
        self.status_filter_combo.currentIndexChanged.connect(self.update_onset_filter)
        filter_layout.addWidget(self.status_filter_combo)
        onset_layout.addLayout(filter_layout)
        nav_layout = QHBoxLayout()
        self.prev_onset_btn = QPushButton("← Prev Onset")
        self.next_onset_btn = QPushButton("Next Onset →")
        self.prev_onset_btn.clicked.connect(self.prev_onset)
        self.next_onset_btn.clicked.connect(self.next_onset)
        nav_layout.addWidget(self.prev_onset_btn)
        nav_layout.addWidget(self.next_onset_btn)
        onset_layout.addLayout(nav_layout)
        self.onset_info_label = QLabel("No onsets loaded")
        onset_layout.addWidget(self.onset_info_label)

        # Offset validation widget (hidden by default) - REFACTORED
        self.setup_offset_validation_widget()  # This will create and configure the new widget
        onset_layout.addWidget(self.offset_validation_widget)

        validation_layout = QHBoxLayout()
        self.accept_btn = QPushButton("✓ Accept")
        self.edit_btn = QPushButton("✎ Edit")
        self.reject_btn = QPushButton("✗ Reject")
        self.change_type_btn = QPushButton("Change Type")
        self.undo_btn = QPushButton("↩ Undo")
        self.accept_btn.clicked.connect(self.start_offset_validation)  # Changed connection
        self.edit_btn.clicked.connect(self.start_edit_onset)
        self.reject_btn.clicked.connect(lambda: self.validate_onset('rejected'))
        self.change_type_btn.clicked.connect(self.show_change_type_dropdown)
        self.undo_btn.clicked.connect(self.undo_last_action)
        validation_layout.addWidget(self.accept_btn)
        validation_layout.addWidget(self.edit_btn)
        validation_layout.addWidget(self.reject_btn)
        validation_layout.addWidget(self.change_type_btn)
        validation_layout.addWidget(self.undo_btn)
        onset_layout.addLayout(validation_layout)

        # Remove shift left/right buttons and their layout
        # Add edit widgets (hidden by default)
        self.edit_widget = QWidget()
        edit_form = QHBoxLayout()
        self.edit_onset_spinbox = QSpinBox()
        self.edit_onset_spinbox.setRange(0, 999999)
        self.edit_offset_spinbox = QSpinBox()
        self.edit_offset_spinbox.setRange(0, 999999)
        edit_form.addWidget(QLabel("Onset to Offset:"))
        edit_form.addWidget(self.edit_onset_spinbox)
        edit_form.addWidget(QLabel("→"))
        edit_form.addWidget(self.edit_offset_spinbox)
        self.finish_edit_btn = QPushButton("Finish Edit")
        self.finish_edit_btn.clicked.connect(self.finish_edit_onset)
        edit_form.addWidget(self.finish_edit_btn)
        self.edit_widget.setLayout(edit_form)
        self.edit_widget.hide()
        onset_layout.addWidget(self.edit_widget)
        # Remove this line - split_widget no longer exists
        # Remove shift left/right buttons and their layout
        # Add edit widgets (hidden by default)
        # (undo_layout will be added at the end, after all widgets)
        onset_group.setLayout(onset_layout)
        # Groupe Save & Export
        save_group = QGroupBox("Save & Export")
        save_layout = QHBoxLayout()
        # Ajout du sélecteur de dossier d'export à gauche
        self.export_path_lineedit = QLineEdit()
        self.export_path_lineedit.setReadOnly(True)
        self.export_path_lineedit.setFixedWidth(220)
        self.export_path_btn = QPushButton("...")
        self.export_path_btn.setFixedWidth(30)
        self.export_path_btn.clicked.connect(self.choose_export_path)
        export_folder_label = QLabel("Export folder:")
        export_folder_label.setContentsMargins(0, 0, 0, 0)
        save_layout.addWidget(export_folder_label)
        save_layout.addWidget(self.export_path_lineedit)
        save_layout.addWidget(self.export_path_btn)
        # Puis le bouton Save and Export à droite
        self.save_export_btn = QPushButton("💾 Save and Export")
        self.save_export_btn.setStyleSheet("QPushButton { font-weight: bold; font-size: 12px; padding: 8px; }")
        self.save_export_btn.setFixedWidth(180)
        self.save_export_btn.clicked.connect(self.export_all_outputs)
        save_layout.addWidget(self.save_export_btn)
        save_group.setLayout(save_layout)
        # Place Performance Score (metrics_group) just below the timeline (drag)
        metrics_group = QGroupBox("Performance Score")
        metrics_layout = QVBoxLayout()
        self.metrics_text = QTextEdit()
        self.metrics_text.setMaximumHeight(80)  # Reduce height in Y
        self.metrics_text.setMinimumHeight(50)
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet("font-size: 10px;")  # Smaller font
        metrics_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_layout)
        left_panel.addWidget(metrics_group)
        left_panel.addWidget(save_group)
        right_panel.addWidget(timeline_group)
        right_panel.addWidget(onset_group)
        right_panel.addWidget(manual_event_group)
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMinimumWidth(300)
        splitter.addWidget(left_widget)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        splitter.addWidget(right_widget)
        splitter.setSizes([850, 750])  # Réactivé pour une séparation visuelle comme avant
        content_layout.addWidget(splitter)
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)
        # Manual-only mode flags (when only raw motion is loaded)
        self.manual_mode_primary = False
        self.manual_mode_secondary = False
        # Supprimer l'appel à QTimer.singleShot(b0, self.finalize_ui) et la méthode finalize_ui
        # Supprimer l'ajout du bouton Pause (et des autres boutons de contrôle vidéo) au left_panel si ce n'est pas nécessaire
        # Garder la configuration des boutons dans la barre de contrôle vidéo
        # S'assurer que Pause reste toujours cliquable et visible

        # Ajout du sélecteur de dossier d'export
        

    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        # Loop playback state
        self.is_looping_event = False
        self.loop_start_frame = None
        self.loop_end_frame = None
        
    def run_notebook(self, motion_energy_file):
        try:
            # Chemin absolu vers le notebook
            notebook_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "demo", "building_state_and_twitch_detection_avg.ipynb")
            )
            if not os.path.exists(notebook_path):
                QMessageBox.critical(self, "Error", f"Notebook not found:\n{notebook_path}")
                return "Notebook not found."
            
            save_dir = os.path.dirname(os.path.abspath(motion_energy_file))
            os.makedirs(save_dir, exist_ok=True)

            # Ouvrir et exécuter le notebook
            with open(notebook_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
            ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})

            return f"Notebook executed successfully with {motion_energy_file}"

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to execute notebook:\n{e}")
            return f"Execution failed: {e}"

    def select_motion_energy(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Motion Energy File", "", "Numpy files (*.npy);;JSON files (*.json);;All Files (*)", options=options)
        
        if file_path:
            # Store path to enable compute button and use it later
            self.motion_energy_path = file_path
            self.compute_motion_energy_btn.setEnabled(True)
            result_message = self.run_notebook(file_path)
            QMessageBox.information(self, "Success", result_message)
        else:
            QMessageBox.warning(self, "No File Selected", "Please select a motion energy file.")

    def run_state_detection_script(self):
        """Run the demo/building_state_and_twitch_detection_avg.py script as-is via subprocess.
        This preserves the current behavior and parameters defined inside the script.
        """
        try:
            import subprocess, sys
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QTextEdit
            # Resolve script path relative to this file
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo", "building_state_and_twitch_detection_avg.py"))
            if not os.path.exists(script_path):
                QMessageBox.critical(self, "Error", f"Analysis script not found:\n{script_path}")
                return

            # Run the script with Python interpreter; set cwd to the demo folder
            cwd = os.path.dirname(script_path)
            # If a motion energy path has been selected/loaded, pass it to the script
            args = [sys.executable, script_path]
            me_path = getattr(self, 'motion_energy_path', None)
            if me_path and os.path.exists(me_path):
                args.append(me_path)

            # Open threshold selection dialog
            class ThresholdDialog(QDialog):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.setWindowTitle("Choose thresholds")
                    layout = QVBoxLayout(self)
                    # Intro/explanation block
                    intro = QLabel(
                        """
                        <b>Threshold Methods</b><br>
                        • <b>Binary threshold</b> (Active vs Rest on smoothed motion energy):<br>
                        &nbsp;&nbsp;- <b>Otsu</b>: best when the histogram is clearly bimodal (two states).<br>
                        &nbsp;&nbsp;- <b>Li</b>: useful for multi-peak or complex histograms.<br>
                        &nbsp;&nbsp;- <b>Mean+SD</b>: simple threshold = mean + standard deviation.<br>
                        • <b>Twitch threshold</b> (detect twitches during resting periods):<br>
                        &nbsp;&nbsp;- <b>Otsu</b>: good when rest distribution appears bimodal.<br>
                        &nbsp;&nbsp;- <b>Li</b>: more permissive; tends to detect more twitches.<br>
                        &nbsp;&nbsp;- <b>MAD</b>: median + 3*MAD (robust to outliers).<br>
                        &nbsp;&nbsp;- <b>95th percentile</b>: keeps only the strongest peaks.<br>
                        &nbsp;&nbsp;- <b>Mean+3*SD</b>: higher threshold, fewer false positives.
                        """
                    )
                    intro.setWordWrap(True)
                    layout.addWidget(intro)
                    settings = QSettings('Mousecraft', 'Thresholds')
                    last_binary = settings.value('binary_method', 'otsu')
                    last_twitch = settings.value('twitch_method', 'otsu')
                    # Binary method
                    bin_layout = QHBoxLayout()
                    bin_layout.addWidget(QLabel("Binary threshold:"))
                    self.binary_combo = QComboBox()
                    self.binary_combo.addItems(["otsu", "li", "mean_sd"])  # must match script options
                    # Tooltips for binary methods
                    self.binary_combo.setItemData(0, "Otsu: optimal separation for bimodal distributions.", Qt.ToolTipRole)
                    self.binary_combo.setItemData(1, "Li: robust when the histogram has multiple peaks.", Qt.ToolTipRole)
                    self.binary_combo.setItemData(2, "Mean+SD: threshold = mean + standard deviation (simple).", Qt.ToolTipRole)
                    try:
                        if last_binary in ["otsu","li","mean_sd"]:
                            self.binary_combo.setCurrentText(last_binary)
                        else:
                            self.binary_combo.setCurrentText("otsu")
                    except Exception:
                        self.binary_combo.setCurrentText("otsu")
                    bin_layout.addWidget(self.binary_combo)
                    layout.addLayout(bin_layout)
                    # Twitch method
                    tw_layout = QHBoxLayout()
                    tw_layout.addWidget(QLabel("Twitch threshold:"))
                    self.twitch_combo = QComboBox()
                    self.twitch_combo.addItems(["otsu", "li", "mad", "percentile_95", "mean_3sd"])  # must match script options
                    # Tooltips for twitch methods
                    self.twitch_combo.setItemData(0, "Otsu: automatic split when bimodality is clear.", Qt.ToolTipRole)
                    self.twitch_combo.setItemData(1, "Li: more permissive; detects more twitches.", Qt.ToolTipRole)
                    self.twitch_combo.setItemData(2, "MAD: median + 3*MAD, robust to outliers.", Qt.ToolTipRole)
                    self.twitch_combo.setItemData(3, "95th percentile: keeps only the strongest peaks.", Qt.ToolTipRole)
                    self.twitch_combo.setItemData(4, "Mean+3*SD: higher threshold, fewer false positives.", Qt.ToolTipRole)
                    try:
                        if last_twitch in ["otsu","li","mad","percentile_95","mean_3sd"]:
                            self.twitch_combo.setCurrentText(last_twitch)
                        else:
                            self.twitch_combo.setCurrentText("otsu")
                    except Exception:
                        self.twitch_combo.setCurrentText("otsu")
                    tw_layout.addWidget(self.twitch_combo)
                    layout.addLayout(tw_layout)
                    # Buttons
                    btns = QHBoxLayout()
                    ok_btn = QPushButton("Run")
                    cancel_btn = QPushButton("Cancel")
                    ok_btn.clicked.connect(self.accept)
                    cancel_btn.clicked.connect(self.reject)
                    btns.addWidget(ok_btn)
                    btns.addWidget(cancel_btn)
                    layout.addLayout(btns)
                    # Default larger size for readability
                    self.resize(720, 420)

            dlg = ThresholdDialog(self)
            if dlg.exec_() != QDialog.Accepted:
                return
            # Append threshold choices as positional args expected by the script
            chosen_binary = dlg.binary_combo.currentText()
            chosen_twitch = dlg.twitch_combo.currentText()
            # Persist choices
            try:
                settings = QSettings('Mousecraft', 'Thresholds')
                settings.setValue('binary_method', chosen_binary)
                settings.setValue('twitch_method', chosen_twitch)
            except Exception:
                pass
            # Add thresholds and no-plots flag (prevents blocking GUI by plt.show)
            args.extend([chosen_binary, chosen_twitch, '--no-plots'])

            # Non-blocking execution with live log dialog
            log_dialog = QDialog(self)
            log_dialog.setWindowTitle("Running analysis...")
            vbox = QVBoxLayout(log_dialog)
            log_view = QTextEdit()
            log_view.setReadOnly(True)
            vbox.addWidget(log_view)
            hbox = QHBoxLayout()
            cancel_btn = QPushButton("Cancel")
            close_btn = QPushButton("Close")
            close_btn.setEnabled(False)
            hbox.addWidget(cancel_btn)
            hbox.addWidget(close_btn)
            vbox.addLayout(hbox)

            proc = QProcess(self)
            proc.setWorkingDirectory(cwd)
            # Capture stdout/stderr
            proc.setProcessChannelMode(QProcess.MergedChannels)

            def _append_output():
                data = proc.readAllStandardOutput().data().decode(errors='ignore')
                if data:
                    log_view.append(data)

            proc.readyReadStandardOutput.connect(_append_output)
            proc.readyReadStandardError.connect(_append_output)

            def _on_finished(exitCode, exitStatus):
                _append_output()
                success = (exitCode == 0)
                loaded_labels = False
                summary_data = None
                if success:
                    try:
                        me_path_local = getattr(self, 'motion_energy_path', None)
                        if me_path_local and os.path.exists(me_path_local):
                            out_dir = os.path.join(os.path.dirname(me_path_local), 'motion_annotation_average')
                            labels_path = os.path.join(out_dir, 'gui_labels.csv')
                            summary_path = os.path.join(out_dir, 'analysis_summary.json')
                            if os.path.exists(labels_path):
                                self.load_framewise_table(labels_path)
                                loaded_labels = True
                            if os.path.exists(summary_path):
                                import json
                                with open(summary_path, 'r', encoding='utf-8') as f:
                                    summary_data = json.load(f)
                    except Exception:
                        loaded_labels = False
                close_btn.setEnabled(True)
                cancel_btn.setEnabled(False)
                if success and loaded_labels:
                    log_view.append("\n✅ Done. gui_labels.csv loaded into the timeline.")
                elif success:
                    log_view.append("\n✅ Done. Outputs saved in motion_annotation_average.")
                else:
                    log_view.append(f"\n❌ Failed with exit code {exitCode}.")

                # Show analysis summary dialog if available
                try:
                    if summary_data:
                        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QGridLayout
                        dlg = QDialog(self)
                        dlg.setWindowTitle("Analysis Summary")
                        v = QVBoxLayout(dlg)
                        grid = QGridLayout()

                        def add_row(r, name, value):
                            grid.addWidget(QLabel(f"{name}:"), r, 0)
                            grid.addWidget(QLabel(str(value)), r, 1)

                        r = 0
                        add_row(r, "Mode", summary_data.get('mode')); r += 1
                        snr = summary_data.get('snr'); add_row(r, "SNR", f"{snr:.3f}" if isinstance(snr, (int,float)) else snr); r += 1
                        smooth = summary_data.get('smoothing', {})
                        add_row(r, "Smooth window", smooth.get('adaptive_window_length')); r += 1
                        add_row(r, "Polyorder", smooth.get('polyorder')); r += 1
                        bim = summary_data.get('bimodality', {})
                        add_row(r, "Bimodal", bim.get('is_bimodal')); r += 1
                        add_row(r, "KDE peaks", bim.get('num_kde_peaks')); r += 1
                        add_row(r, "Dip p-value", bim.get('dip_p_value')); r += 1
                        th = summary_data.get('thresholds', {})
                        bin_th = th.get('binary', {})
                        add_row(r, "Binary method", bin_th.get('method')); r += 1
                        add_row(r, "Binary value", bin_th.get('value')); r += 1
                        tw_th = th.get('twitch', {})
                        add_row(r, "Twitch method", tw_th.get('method')); r += 1
                        add_row(r, "Twitch value", tw_th.get('value')); r += 1
                        cnt = summary_data.get('counts', {})
                        add_row(r, "# Active motions", cnt.get('n_active_motions')); r += 1
                        add_row(r, "# Twitches", cnt.get('n_twitches')); r += 1
                        v.addLayout(grid)
                        dlg.exec_()
                except Exception:
                    pass

            def _on_cancel():
                try:
                    proc.kill()
                except Exception:
                    pass
                cancel_btn.setEnabled(False)

            cancel_btn.clicked.connect(_on_cancel)
            close_btn.clicked.connect(log_dialog.accept)
            proc.finished.connect(_on_finished)

            # Start process
            program = sys.executable
            proc.start(program, args[1:])
            log_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run analysis script:\n{e}")

    def load_motion_energy(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Motion Energy', '', 'CSV/Excel (*.csv *.xlsx *.npy)')
        if fname:
            # Remember path and enable compute button for script
            self.motion_energy_path = fname if fname.lower().endswith('.npy') else None
            if self.motion_energy_path:
                self.compute_motion_energy_btn.setEnabled(True)
            # Existing behavior retained below
            import os
            parent = os.path.dirname(fname)
            grandparent = os.path.dirname(parent)
            arriere_grandparent = os.path.dirname(grandparent)
            grandparent_folder = os.path.basename(grandparent)
            arriere_grandparent_folder = os.path.basename(arriere_grandparent)
            if fname.endswith('.npy'):
                self.motion_energy = np.load(fname)
            else:
                df = pd.read_excel(fname) if fname.endswith('.xlsx') else pd.read_csv(fname)
                self.motion_energy = df.select_dtypes(include=[np.number]).iloc[:, 0].values
            self.prepare_motion_energy()
            self.timeline_canvas.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, getattr(self, 'event_status', None))
            self.reset_zoom_timeline()
            self.reset_video_zoom()
            self.maybe_start_auto_save()
        
    def load_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', 'Videos (*.avi *.mp4 *.mov *.mkv *.tiff *.tif)')
        if fname:
            self.video_path = fname
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open video file")
                return
                
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # Ne pas écraser le FPS choisi
            # self.fps_lineedit.setText("")  # Ne pas vider le champ FPS
            self.frame_slider.setMaximum(self.total_frames - 1)
            # Aller au premier onset si disponible
            if hasattr(self, 'onsets') and self.onsets:
                self.current_frame = self.onsets[0]
            else:
                self.current_frame = 0
            self.show_frame(self.current_frame)
            self.update_frame_info()
            self.frame_slider.setValue(self.current_frame)
            
            # Update onset/offset ranges
            self.onset_spinbox.setMaximum(self.total_frames - 1)
            self.offset_spinbox.setMaximum(self.total_frames - 1)
            # Set video folder label to grandparent and parent folder name
            import os
            parent_folder = os.path.basename(os.path.dirname(fname))
            grandparent_folder = os.path.basename(os.path.dirname(os.path.dirname(fname)))
            self.video_folder_label.setText(f"Video folder : {grandparent_folder} / {parent_folder}")
            self.maybe_start_auto_save()

    def add_camera(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Second Video', '', 'Videos (*.avi *.mp4 *.mov *.mkv *.tiff *.tif)')
        if not fname:
            return
        self.second_video_path = fname
        self.cap2 = cv2.VideoCapture(self.second_video_path)
        if not self.cap2.isOpened():
            QMessageBox.critical(self, "Error", "Could not open second video file")
            self.cap2 = None
            return
        # Show second camera label and set equal share between cameras without changing overall panel layout
        self.second_video_label.show()
        # Allow the first camera to shrink to half by removing its large minimum size
        try:
            self.video_label.setMinimumSize(0, 0)
            self.second_video_label.setMinimumSize(0, 0)
        except Exception:
            pass
        self.video_stack_layout.setStretch(0, 1)
        self.video_stack_layout.setStretch(1, 1)
        # Reset second camera zoom so it is not oddly zoomed on add
        self.reset_video_zoom(target=2)
        # Trigger a repaint at current frame for both cameras
        self.show_frame(self.current_frame)
        
    def load_motion_energy(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Motion Energy', '', 'CSV/Excel (*.csv *.xlsx *.npy)')
        if fname:
            # Set motion energy folder label to arrière-grand-parent and grandparent folder name
            import os
            parent = os.path.dirname(fname)
            grandparent = os.path.dirname(parent)
            arriere_grandparent = os.path.dirname(grandparent)
            grandparent_folder = os.path.basename(grandparent)
            arriere_grandparent_folder = os.path.basename(arriere_grandparent)
            self.motion_energy_folder_label.setText(f"Motion energy folder : {arriere_grandparent_folder} / {grandparent_folder}")
            if fname.endswith('.npy'):
                self.motion_energy = np.load(fname)
            elif fname.endswith('.csv'):
                df = pd.read_csv(fname)
                self.motion_energy = df.select_dtypes(include=[np.number]).iloc[:, 0].values
            else:
                df = pd.read_excel(fname)
                self.motion_energy = df.select_dtypes(include=[np.number]).iloc[:, 0].values
                
            # Pad to divisible by 5 and average
            self.prepare_motion_energy()
            # Reset zoom when loading motion energy directly
            self.timeline_canvas.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, getattr(self, 'event_status', None))
            self.reset_zoom_timeline()
            self.reset_video_zoom()
            self.maybe_start_auto_save()
            
    def prepare_motion_energy(self):
        """Pad motion energy to divisible by 5 and average"""
        if self.motion_energy is None:
            return
        # Ensure numpy float64 array
        self.motion_energy = np.asarray(self.motion_energy, dtype=np.float64)
        # Pad to make divisible by 5
        remainder = len(self.motion_energy) % 5
        if remainder != 0:
            padding_needed = 5 - remainder
            self.motion_energy = np.pad(self.motion_energy, (0, padding_needed), mode='edge')
        # Reshape and average
        new_length = len(self.motion_energy) // 5
        self.motion_energy = self.motion_energy.reshape(new_length, 5).mean(axis=1)
        # Update total frames
        self.total_frames = len(self.motion_energy)
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.onset_spinbox.setMaximum(self.total_frames - 1)
        self.offset_spinbox.setMaximum(self.total_frames - 1)
            
    def load_classifications(self):
        try:
            fname, _ = QFileDialog.getOpenFileName(self, 'Load Classifications', '', 'JSON/CSV/Excel (*.json *.csv *.xlsx)')
            if fname:
                import os
                parent = os.path.dirname(fname)
                grandparent = os.path.dirname(parent)
                arriere_grandparent = os.path.dirname(grandparent)
                grandparent_folder = os.path.basename(grandparent)
                arriere_grandparent_folder = os.path.basename(arriere_grandparent)
                self.motion_energy_folder_label.setText(f"Classification folder : {arriere_grandparent_folder} / {grandparent_folder}")
                if fname.endswith('.json'):
                    self.load_json_classifications(fname)
                else:
                    import pandas as pd
                    loaded_df = pd.read_csv(fname) if fname.endswith('.csv') else pd.read_excel(fname)
                    self.input_classification_df = loaded_df.copy()  # Garde une copie immuable pour l'export
                    cols = set(loaded_df.columns)
                    if {'active', 'twitch'}.issubset(cols):
                        self.load_framewise_table(fname)
                    elif {'active_motion_onset', 'active_motion_offset', 'twitch_onset', 'twitch_offset'}.issubset(cols):
                        self.load_excel_classifications(fname)
                    else:
                        QMessageBox.warning(self, "Warning", "File format not recognized. Please provide a valid classification file.")
                self.undo_btn.setEnabled(True)
                # Reset zoom when loading classifications directly
                self.reset_zoom_timeline()
                self.reset_video_zoom()
                self.maybe_start_auto_save()
                # Reset mouse milestone state
                self._mouse_half_shown = False
                self._mouse_milestones = set()
                self._mouse_milestones_shown = set()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load classifications: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_input(self):
        """Unified loader for motion energy or classifications (.json/.csv/.xlsx/.npy).
        If a motion energy is already loaded, this can be used to add a second input as well.
        """
        fname, _ = QFileDialog.getOpenFileName(
            self,
            'Load an input',
            '',
            'All Supported (*.json *.csv *.xlsx *.npy);;JSON (*.json);;CSV (*.csv);;Excel (*.xlsx);;NumPy (*.npy)'
        )
        if not fname:
            return
        try:
            lower = fname.lower()
            if lower.endswith('.npy'):
                # Treat as motion energy (manual-only mode)
                self.motion_energy = np.load(fname)
                # Remember path and enable compute button for external script
                self.motion_energy_path = fname
                if hasattr(self, 'compute_motion_energy_btn'):
                    self.compute_motion_energy_btn.setEnabled(True)
                # Prepare and display
                self.prepare_motion_energy()
                self.timeline_canvas.plot_motion_energy(
                    self.motion_energy,
                    self.onsets,
                    self.onset_types,
                    self.timeline_canvas.event_offsets,
                    getattr(self, 'event_status', None)
                )
                self.reset_zoom_timeline()
                self.manual_mode_primary = True
            elif lower.endswith('.csv') or lower.endswith('.xlsx'):
                # Try as classifications first; if not recognized, fall back to motion energy (first numeric column)
                try:
                    self.load_classifications_from_path(fname)
                except Exception:
                    # Fallback: motion energy from first numeric column
                    df = pd.read_csv(fname) if lower.endswith('.csv') else pd.read_excel(fname)
                    self.motion_energy = df.select_dtypes(include=[np.number]).iloc[:, 0].values
                    self.prepare_motion_energy()
                    self.timeline_canvas.plot_motion_energy(
                        self.motion_energy,
                        self.onsets,
                        self.onset_types,
                        self.timeline_canvas.event_offsets,
                        getattr(self, 'event_status', None)
                    )
                    self.reset_zoom_timeline()
                    self.manual_mode_primary = True
            else:
                QMessageBox.warning(self, 'Unsupported file', 'Please select a .json, .csv, .xlsx, or .npy file.')
                return
            # Reset video zoom on input load (if videos are present)
            self.reset_video_zoom()
            self.maybe_start_auto_save()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load input: {str(e)}')
            import traceback
            traceback.print_exc()
            
    def load_second_input(self):
        """Load a second motion input and display it on the secondary timeline without changing overall layout sizing."""
        fname, _ = QFileDialog.getOpenFileName(
            self, 'Load second input', '',
            'All Supported (*.json *.csv *.xlsx *.npy);;JSON (*.json);;CSV (*.csv);;Excel (*.xlsx);;NumPy (*.npy)'
        )
        if not fname:
            return
        try:
            lower = fname.lower()
            classified_loaded = False

            if lower.endswith('.npy'):
            # Motion energy only
                second_motion = np.load(fname)
                self.timeline_canvas2.motion_energy = second_motion
                self.timeline_canvas2.onsets = []
                self.timeline_canvas2.onset_types = {}
                self.timeline_canvas2.event_offsets = {}
                self.timeline_canvas2.onset_validations = {}
                self.manual_mode_secondary = True
                classified_loaded = False

            elif lower.endswith('.json'):
                # JSON classifications (events only, no energy)
                self.onsets2, self.onset_types2, self.timeline_canvas2.event_offsets = self.load_json_classifications(fname)
                # Ensure motion energy exists for secondary; fallback to primary length or inferred length
                if getattr(self.timeline_canvas2, 'motion_energy', None) is None:
                    try:
                        if getattr(self, 'motion_energy', None) is not None:
                            length = len(self.motion_energy)
                        else:
                            max_frame = 0
                            if self.onsets2:
                                max_frame = max(max_frame, max(self.onsets2))
                            if getattr(self.timeline_canvas2, 'event_offsets', {}):
                                max_frame = max(max_frame, max(self.timeline_canvas2.event_offsets.values()))
                            length = max_frame + 1 if max_frame > 0 else 1000
                        self.timeline_canvas2.motion_energy = np.zeros(length, dtype=float)
                    except Exception:
                        pass
                classified_loaded = True

            elif lower.endswith('.csv') or lower.endswith('.xlsx'):
                # Try as classifications
                try:
                    # Load as classifications for secondary
                    self.onsets2, self.onset_types2, self.timeline_canvas2.event_offsets = self.load_classifications_from_path(fname)
                    # Also extract motion energy for secondary if available
                    try:
                        if lower.endswith('.csv'):
                            df2 = pd.read_csv(fname)
                        else:
                            df2 = pd.read_excel(fname)
                        if 'motion_energy' in df2.columns:
                            self.timeline_canvas2.motion_energy = df2['motion_energy'].values.astype(np.float64)
                    except Exception:
                        pass
                    classified_loaded = True
                except Exception:
                    # fallback: motion-only
                    if lower.endswith('.csv'):
                        df = pd.read_csv(fname)
                    else:
                        df = pd.read_excel(fname)
                    second_motion = df.select_dtypes(include=[np.number]).iloc[:, 0].values
                    self.timeline_canvas2.motion_energy = second_motion
                    self.timeline_canvas2.onsets = []
                    self.timeline_canvas2.onset_types = {}
                    self.timeline_canvas2.event_offsets = {}
                    self.timeline_canvas2.onset_validations = {}
                    self.manual_mode_secondary = True
                    classified_loaded = False

            else:
                QMessageBox.warning(
                    self, 'Unsupported file',
                    'Please select a .json, .csv, .xlsx, or .npy file for the second input.'
                )
                return

            # Push parsed second annotations into canvas2
            if classified_loaded:
                self.timeline_canvas2.onsets = list(self.onsets2)
                self.timeline_canvas2.onset_types = dict(self.onset_types2)
                self.timeline_canvas2.onset_validations = {o: 'pending' for o in self.onsets2}
                self.manual_mode_secondary = False
                self.timeline_canvas2.visible_event_types = getattr(
                    self.timeline_canvas, 'visible_event_types', ['twitch', 'active']
                )

            # Prepare and plot
            if not classified_loaded:
                self.timeline_canvas2.onsets = []
                self.timeline_canvas2.onset_types = {}
                self.timeline_canvas2.event_offsets = {}
                self.timeline_canvas2.onset_validations = {}
            else:
                # Ensure total_frames and motion_energy are set for secondary
                if getattr(self.timeline_canvas2, 'motion_energy', None) is None:
                    # Fallback to primary length
                    default_len = len(self.motion_energy) if getattr(self, 'motion_energy', None) is not None else 1000
                    self.timeline_canvas2.motion_energy = np.zeros(default_len, dtype=float)
                self.timeline_canvas2.total_frames = len(self.timeline_canvas2.motion_energy)

            self.timeline_canvas2.current_frame = getattr(self.timeline_canvas, 'current_frame', 0)
            # For initial render of input 2, do NOT preserve old view; reset to full extent
            # Ensure motion_energy exists
            if getattr(self.timeline_canvas2, 'motion_energy', None) is None:
                default_len = len(self.motion_energy) if getattr(self, 'motion_energy', None) is not None else 1000
                self.timeline_canvas2.motion_energy = np.zeros(default_len, dtype=float)
            self.timeline_canvas2.total_frames = len(self.timeline_canvas2.motion_energy)
            self.timeline_canvas2.plot_motion_energy(
                self.timeline_canvas2.motion_energy,
                getattr(self, 'onsets2', []),
                getattr(self, 'onset_types2', {}),
                getattr(self.timeline_canvas2, 'event_offsets', {}),
                None
            )

            # Show second timeline and equalize space
            self.timeline_canvas2.show()
            self.timeline_splitter.setSizes([1, 1])
            # Force both timelines to full-extent view and synchronize xlim
            self.reset_zoom_timeline()

            # Enable radio to select second input for annotation
            if hasattr(self, 'annotate_secondary_radio'):
                self.annotate_secondary_radio.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load second input: {str(e)}')
            import traceback
            traceback.print_exc()

    def load_classifications_from_path(self, fname):
        """Helper to load classifications from a known good CSV/XLSX path."""
        if hasattr(self, 'annotate_secondary_radio'):
            self.annotate_secondary_radio.setEnabled(True)

    def load_classifications_from_path(self, fname):
        """Helper to load classifications from a known good CSV/XLSX path."""
        import os
        parent = os.path.dirname(fname)
        grandparent = os.path.dirname(parent)
        arriere_grandparent = os.path.dirname(grandparent)
        grandparent_folder = os.path.basename(grandparent)
        arriere_grandparent_folder = os.path.basename(arriere_grandparent)
        self.motion_energy_folder_label.setText(f"Classification folder : {arriere_grandparent_folder} / {grandparent_folder}")
        loaded_df = pd.read_csv(fname) if fname.endswith('.csv') else pd.read_excel(fname)
        self.input_classification_df = loaded_df.copy()
        cols = set(loaded_df.columns)
        if {'active', 'twitch'}.issubset(cols):
            self.load_framewise_table(fname)
        elif {'active_motion_onset', 'active_motion_offset', 'twitch_onset', 'twitch_offset'}.issubset(cols):
            self.load_excel_classifications(fname)
        else:
            # Generic dynamic schema: each non-motion_energy, non-score/status column is treated as a binary mask
            df = loaded_df
            if 'motion_energy' in df.columns and self.motion_energy is None:
                self.motion_energy = df['motion_energy'].values.astype(np.float64)
                self.prepare_motion_energy()
            self.onsets = []
            self.onset_types = {}
            self.timeline_canvas.event_offsets = {}
            self.event_status = {}
            self.curated_events = {}
            self.timeline_canvas.onset_validations = {}
            # Identify candidate event columns
            ignore_cols = {'motion_energy', 'status', 'score', 'frame_idx'}
            candidate_cols = [c for c in df.columns if c not in ignore_cols]
            created = 0
            for col in candidate_cols:
                etype = str(col).lower().strip()
                if etype not in self.available_event_types:
                    self.available_event_types.append(etype)
                if etype not in self.event_type_colors:
                    used = {c.lower() for c in self.event_type_colors.values()}
                    picked = None
                    for c in self._auto_palette:
                        if c.lower() not in used:
                            picked = c
                            break
                    if picked is None:
                        picked = f"#{hash(etype) & 0xFFFFFF:06x}"
                    self.event_type_colors[etype] = picked
                arr = df[col].astype(int).values
                in_event = False
                onset = None
                for i, val in enumerate(arr):
                    if val == 1 and not in_event:
                        onset = i
                        in_event = True
                    elif (val == 0 or i == len(arr) - 1) and in_event:
                        offset = i - 1 if val == 0 else i
                        if onset not in self.onsets:
                            self.onsets.append(onset)
                            self.onset_types[onset] = etype
                            self.timeline_canvas.event_offsets[onset] = offset
                        # Optional status/score if present
                        status = str(df.at[onset, 'status']).strip().lower() if 'status' in df.columns else 'pending'
                        score = df.at[onset, 'score'] if 'score' in df.columns else 0
                        if status == 'edited':
                            gui_status = 'edited'
                            gui_score = 1 if float(score) == 1 else 0.5
                        elif status == 'accepted':
                            gui_status = 'accepted'
                            gui_score = 1
                        elif status == 'rejected':
                            gui_status = 'rejected'
                            gui_score = -1
                        elif status == 'manually added':
                            gui_status = 'manually added'
                            gui_score = 0
                        else:
                            gui_status = 'pending'
                            gui_score = 0
                        self.timeline_canvas.onset_validations[onset] = gui_status
                        self.event_status[onset] = gui_score
                        if etype not in self.curated_events:
                            self.curated_events[etype] = []
                        self.curated_events[etype].append([onset, offset, 1])
                        in_event = False
                        created += 1
                # If an event is still open at the very end, close it
                if in_event and onset is not None:
                    offset = len(arr) - 1
                    if onset not in self.onsets:
                        self.onsets.append(onset)
                        self.onset_types[onset] = etype
                        self.timeline_canvas.event_offsets[onset] = offset
                    status = str(df.at[onset, 'status']).strip().lower() if 'status' in df.columns else 'pending'
                    score = df.at[onset, 'score'] if 'score' in df.columns else 0
                    if status == 'edited':
                        gui_status = 'edited'
                        gui_score = 1 if float(score) == 1 else 0.5
                    elif status == 'accepted':
                        gui_status = 'accepted'
                        gui_score = 1
                    elif status == 'rejected':
                        gui_status = 'rejected'
                        gui_score = -1
                    elif status == 'manually added':
                        gui_status = 'manually added'
                        gui_score = 0
                    else:
                        gui_status = 'pending'
                        gui_score = 0
                    self.timeline_canvas.onset_validations[onset] = gui_status
                    self.event_status[onset] = gui_score
                    if etype not in self.curated_events:
                        self.curated_events[etype] = []
                    self.curated_events[etype].append([onset, offset, 1])
                    in_event = False
                    created += 1
            try:
                print(f"[Framewise Loader] Built {created} events for type '{etype}'")
            except Exception:
                pass
            self.onsets = sorted(set(self.onsets))
            # Propagate UI and canvas settings
            if hasattr(self, 'event_type_combo'):
                placeholder = self.event_type_combo.itemText(0) if self.event_type_combo.count() > 0 else "Select type…"
                self.event_type_combo.clear()
                self.event_type_combo.addItem(placeholder)
                self.event_type_combo.addItems(self.available_event_types)
                self.event_type_combo.setCurrentIndex(0)
            if hasattr(self, 'onset_filter_combo'):
                current_idx = self.onset_filter_combo.currentIndex() if self.onset_filter_combo.count() > 0 else 0
                self.onset_filter_combo.clear()
                self.onset_filter_combo.addItems(["All"] + [t.capitalize() for t in self.available_event_types])
                if current_idx < self.onset_filter_combo.count():
                    self.onset_filter_combo.setCurrentIndex(current_idx)
            if hasattr(self, 'change_type_dropdown') and self.change_type_dropdown is not None:
                self.change_type_dropdown.clear()
                self.change_type_dropdown.addItems(self.available_event_types)
            if hasattr(self, 'timeline_canvas'):
                self.timeline_canvas.event_type_colors = dict(self.event_type_colors)
                if getattr(self.timeline_canvas, 'visible_event_types', None) is not None:
                    for t in self.available_event_types:
                        if t not in self.timeline_canvas.visible_event_types:
                            self.timeline_canvas.visible_event_types.append(t)
            if hasattr(self, 'timeline_canvas2'):
                self.timeline_canvas2.event_type_colors = dict(self.event_type_colors)
                if getattr(self.timeline_canvas2, 'visible_event_types', None) is not None:
                    for t in self.available_event_types:
                        if t not in self.timeline_canvas2.visible_event_types:
                            self.timeline_canvas2.visible_event_types.append(t)
            # Update timeline
            self.timeline_canvas.plot_motion_energy_preserve_view(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, getattr(self, 'event_status', None))
            if self.onsets:
                self.goto_onset(0)
        
    def load_json_classifications(self, fname):
        """Load classifications from JSON format"""
        with open(fname, 'r') as f:
            self.classified_events = json.load(f)
            
        # Process classifications
        self.curated_events = {k: [list(pair) for pair in v] for k, v in self.classified_events.items()}
        
        # Extract onsets and their types
        self.onsets = []
        self.onset_types = {}
        
        for event_type, events in self.curated_events.items():
            etype = str(event_type).lower().strip()
            # Register unseen types: add to available list, assign color, and propagate to canvases
            if etype not in self.available_event_types:
                self.available_event_types.append(etype)
                # Assign a color from palette if not present
                if etype not in self.event_type_colors:
                    # Pick next palette color not already used
                    used = {c.lower() for c in self.event_type_colors.values()}
                    picked = None
                    for c in self._auto_palette:
                        if c.lower() not in used:
                            picked = c
                            break
                    if picked is None:
                        picked = f"#{hash(etype) & 0xFFFFFF:06x}"
                    self.event_type_colors[etype] = picked
                # Update UI combos if available
                if hasattr(self, 'event_type_combo'):
                    current_placeholder = self.event_type_combo.itemText(0) if self.event_type_combo.count() > 0 else "Select type…"
                    self.event_type_combo.clear()
                    self.event_type_combo.addItem(current_placeholder)
                    self.event_type_combo.addItems(self.available_event_types)
                    self.event_type_combo.setCurrentIndex(0)
                if hasattr(self, 'onset_filter_combo'):
                    current_idx = self.onset_filter_combo.currentIndex() if self.onset_filter_combo.count() > 0 else 0
                    self.onset_filter_combo.clear()
                    self.onset_filter_combo.addItems(["All"] + [t.capitalize() for t in self.available_event_types])
                    if current_idx < self.onset_filter_combo.count():
                        self.onset_filter_combo.setCurrentIndex(current_idx)
                if hasattr(self, 'change_type_dropdown') and self.change_type_dropdown is not None:
                    self.change_type_dropdown.clear()
                    self.change_type_dropdown.addItems(self.available_event_types)
                # Propagate colors and visibility to canvases
                if hasattr(self, 'timeline_canvas'):
                    self.timeline_canvas.event_type_colors = dict(self.event_type_colors)
                    if getattr(self.timeline_canvas, 'visible_event_types', None) is not None and etype not in self.timeline_canvas.visible_event_types:
                        self.timeline_canvas.visible_event_types.append(etype)
                if hasattr(self, 'timeline_canvas2'):
                    self.timeline_canvas2.event_type_colors = dict(self.event_type_colors)
                    if getattr(self.timeline_canvas2, 'visible_event_types', None) is not None and etype not in self.timeline_canvas2.visible_event_types:
                        self.timeline_canvas2.visible_event_types.append(etype)

            for onset, _ in events:
                self.onsets.append(onset)
                self.onset_types[onset] = etype
                self.original_onsets[onset] = onset  # Track original
                self.original_offsets[onset] = onset  # Track original offset
                
        self.onsets = sorted(self.onsets)
        self.current_onset_idx = 0
        
        # Update timeline
        self.timeline_canvas.plot_motion_energy_preserve_view(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, getattr(self, 'event_status', None))
        
        # Go to first onset
        if self.onsets:
            self.goto_onset(0)
        self.undo_btn.setEnabled(True)
        # Reset mouse milestone state
        self._mouse_half_shown = False
        self._mouse_milestones = set()
        self._mouse_milestones_shown = set()
        
    def load_excel_classifications(self, fname):
        """Load classifications from Excel/CSV with dynamic number of event types.
        Expects pairs of columns <etype>_onset and <etype>_offset (optionally with _motion prefix).
        """
        try:
            if fname.endswith('.csv'):
                df = pd.read_csv(fname)
            else:
                df = pd.read_excel(fname)

            # Load or validate motion energy if present
            if 'motion_energy' in df.columns:
                if self.motion_energy is None:
                    self.motion_energy = df['motion_energy'].values.astype(np.float64)
                    self.prepare_motion_energy()
                else:
                    if len(df) != len(self.motion_energy):
                        QMessageBox.warning(self, "Warning", f"Motion energy length mismatch: CSV has {len(df)} frames, loaded motion energy has {len(self.motion_energy)} frames")
                        return

            # Reset structures
            self.onsets = []
            self.onset_types = {}
            self.timeline_canvas.event_offsets = {}
            self.event_status = {}

            # Discover dynamic etypes from *_onset/*_offset pairs
            cols = set(df.columns)
            base_names = []
            for col in cols:
                if col.endswith('_onset') and col[:-6] + '_offset' in cols:
                    base_names.append(col[:-6])

            # Normalize etype name: remove trailing _motion if present
            def normalize(base: str) -> str:
                et = base
                if et.endswith('_motion'):
                    et = et[:-7]
                return et.lower().strip()

            discovered_types = []
            for base in base_names:
                onset_col = base + '_onset'
                offset_col = base + '_offset'
                etype = normalize(base)
                discovered_types.append(etype)

                # Register new types and assign colors
                if etype not in self.available_event_types:
                    self.available_event_types.append(etype)
                if etype not in self.event_type_colors:
                    used = {c.lower() for c in self.event_type_colors.values()}
                    picked = None
                    for c in self._auto_palette:
                        if c.lower() not in used:
                            picked = c
                            break
                    if picked is None:
                        picked = f"#{hash(etype) & 0xFFFFFF:06x}"
                    self.event_type_colors[etype] = picked

                et_onsets = np.where(df[onset_col] == 1)[0]
                et_offsets = np.where(df[offset_col] == 1)[0]
                for onset in et_onsets:
                    offset = self.find_corresponding_offset(onset, et_offsets)
                    offset = offset + 1 if offset > onset else onset
                    self.onsets.append(onset)
                    self.onset_types[onset] = etype
                    self.timeline_canvas.event_offsets[onset] = offset
                    self.event_status[onset] = 1
                    self.original_onsets[onset] = onset
                    self.original_offsets[onset] = offset

            # Sort
            self.onsets = sorted(self.onsets)
            self.current_onset_idx = 0

            # Update UI combos with discovered types
            if hasattr(self, 'event_type_combo'):
                placeholder = self.event_type_combo.itemText(0) if self.event_type_combo.count() > 0 else "Select type…"
                self.event_type_combo.clear()
                self.event_type_combo.addItem(placeholder)
                self.event_type_combo.addItems(self.available_event_types)
                self.event_type_combo.setCurrentIndex(0)
            if hasattr(self, 'onset_filter_combo'):
                current_idx = self.onset_filter_combo.currentIndex() if self.onset_filter_combo.count() > 0 else 0
                self.onset_filter_combo.clear()
                self.onset_filter_combo.addItems(["All"] + [t.capitalize() for t in self.available_event_types])
                if current_idx < self.onset_filter_combo.count():
                    self.onset_filter_combo.setCurrentIndex(current_idx)
            if hasattr(self, 'change_type_dropdown') and self.change_type_dropdown is not None:
                self.change_type_dropdown.clear()
                self.change_type_dropdown.addItems(self.available_event_types)

            # Propagate colors and visibility to canvases
            if hasattr(self, 'timeline_canvas'):
                self.timeline_canvas.event_type_colors = dict(self.event_type_colors)
                if getattr(self.timeline_canvas, 'visible_event_types', None) is not None:
                    for t in self.available_event_types:
                        if t not in self.timeline_canvas.visible_event_types:
                            self.timeline_canvas.visible_event_types.append(t)
            if hasattr(self, 'timeline_canvas2'):
                self.timeline_canvas2.event_type_colors = dict(self.event_type_colors)
                if getattr(self.timeline_canvas2, 'visible_event_types', None) is not None:
                    for t in self.available_event_types:
                        if t not in self.timeline_canvas2.visible_event_types:
                            self.timeline_canvas2.visible_event_types.append(t)

            # Update timeline
            self.timeline_canvas.plot_motion_energy_preserve_view(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, getattr(self, 'event_status', None))

            # Go to first onset
            if self.onsets:
                self.goto_onset(0)

            QMessageBox.information(self, "Success", f"Loaded {len(self.onsets)} events from {fname}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV file: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def find_corresponding_offset(self, onset, offset_indices):
        """Find the corresponding offset for a given onset"""
        # Find the next offset after the onset
        next_offsets = offset_indices[offset_indices > onset]
        if len(next_offsets) > 0:
            return next_offsets[0]
        else:
            # If no offset found, use onset + 1 frame
            return onset + 1
        
    def set_current_as_onset(self):
        """Set current frame as onset"""
        self.onset_spinbox.setValue(self.current_frame)
        
    def set_current_as_offset(self):
        """Set current frame as offset (exclusive)"""
        self.offset_spinbox.setValue(self.current_frame)
        
    def update_offset_range(self):
        """Update offset range to be > onset"""
        onset = self.onset_spinbox.value()
        self.offset_spinbox.setMinimum(onset + 1)
        # Only set default offset if it's invalid (<= onset)
        current_offset = self.offset_spinbox.value()
        if current_offset <= onset:
            self.offset_spinbox.setValue(onset + 1)
            
    def update_edit_threshold(self):
        """Update the edit threshold value"""
        self.edit_threshold = self.edit_threshold_spinbox.value()
            
    def add_manual_event(self):
        self.unsaved_changes = True
        onset = self.onset_spinbox.value()
        offset_exclusive = self.offset_spinbox.value()
        event_type = self.event_type_combo.currentText()
        if event_type == "Select type…":
            QMessageBox.warning(self, "Invalid Event Type", "Please select an event type (twitch, active, or complex) before adding the event.")
            return
        if onset >= offset_exclusive:
            QMessageBox.warning(self, "Invalid Event", 
                f"Offset frame ({offset_exclusive}) must be greater than onset frame ({onset}).\n\n"
                f"An event must have a duration of at least 1 frame.\n"
                f"Please set the offset to a frame number higher than {onset}.")
            return
        # Choose target store: primary (1) or secondary (2)
        target_secondary = hasattr(self, 'timeline_canvas2') and self.timeline_canvas2.isVisible() and self.active_timeline_index == 2
        if target_secondary:
            target_onsets = self.onsets2
            target_types = self.onset_types2
            target_offsets = self.timeline_canvas2.event_offsets
        else:
            target_onsets = self.onsets
            target_types = self.onset_types
            target_offsets = self.timeline_canvas.event_offsets
        if onset in target_onsets:
            QMessageBox.warning(self, "Warning", f"Event at frame {onset} already exists")
            return

        offset_inclusive = offset_exclusive - 1

        # Check for overlap with existing events in the selected target only (ignore rejected)
        overlapping = []
        for other_onset in target_onsets:
            validations = self.timeline_canvas2.onset_validations if target_secondary else self.timeline_canvas.onset_validations
            if validations.get(other_onset, 'pending') == 'rejected':
                continue
            other_offset = target_offsets.get(other_onset, other_onset)
            if (onset <= other_offset and offset_inclusive >= other_onset) and not (onset < other_onset and offset_inclusive > other_offset):
                overlapping.append((other_onset, other_offset, target_types.get(other_onset, 'unknown')))
        if overlapping:
            msg = "This event overlaps with existing events:\n\n"
            for o, off, typ in overlapping:
                msg += f"- {typ} ({o}-{off})\n"
            msg += "\nDo you want to add it anyway?"
            reply = QMessageBox.question(self, "Overlap Detected", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            
        # Check for total overlap and prompt BEFORE adding, exclude itself
        if target_secondary:
            # Use secondary store for total-overlap check
            self.check_total_overlap_and_prompt_for(
                onsets_list=self.onsets2,
                offsets_map=self.timeline_canvas2.event_offsets,
                onset_types=self.onset_types2,
                validations=getattr(self.timeline_canvas2, 'onset_validations', {}),
                new_onset=onset,
                new_offset=offset_inclusive,
                exclude_onsets=onset
            )
        else:
            self.check_total_overlap_and_prompt(onset, offset_inclusive, exclude_onsets=onset)
        # Ajoute aux structures pour la timeline ciblée
        target_onsets.append(onset)
        target_types[onset] = event_type
        target_offsets[onset] = offset_inclusive
        # Set validation status to manually added
        if target_secondary:
            if not hasattr(self.timeline_canvas2, 'onset_validations'):
                self.timeline_canvas2.onset_validations = {}
            self.timeline_canvas2.onset_validations[onset] = 'manually added'
        else:
            self.timeline_canvas.onset_validations[onset] = 'manually added'
        # Update curated events (primary only, keep existing behavior)
        if not target_secondary:
            if event_type not in self.curated_events:
                self.curated_events[event_type] = []
            self.curated_events[event_type].append([onset, offset_inclusive, 0])
        # Sort onsets to maintain chronological order
        if target_secondary:
            self.onsets2 = sorted(self.onsets2)
            self.filtered_onsets2 = self.onsets2.copy()
        else:
            self.onsets = sorted(self.onsets)
            self.filtered_onsets = self.onsets.copy()
        # Store in undo stack
        self.undo_stack.append(('add_manual_2' if target_secondary else 'add_manual', onset, event_type, offset_inclusive))
        # Redraw timeline and refresh filtered onsets/counter immediately
        self.update_onset_filter()
        self.update_onset_info()
        if target_secondary:
            # Only redraw second timeline
            self.timeline_canvas2.plot_motion_energy_preserve_view(self.timeline_canvas2.motion_energy, self.onsets2, self.onset_types2, self.timeline_canvas2.event_offsets, None)
        else:
            self.redraw()
        # Navigation logic after adding:
        if overlapping:
            # Go to the first overlapped onset
            overlapped_onset = overlapping[0][0]
            if hasattr(self, 'filtered_onsets') and self.filtered_onsets and overlapped_onset in self.filtered_onsets:
                idx = self.filtered_onsets.index(overlapped_onset)
                self.goto_onset(idx)
            elif overlapped_onset in self.onsets:
                idx = self.onsets.index(overlapped_onset)
                self.goto_onset(idx)
        else:
            # Go to the next event in the filtered list after adding
            if not target_secondary and hasattr(self, 'filtered_onsets') and self.filtered_onsets:
                if onset in self.filtered_onsets:
                    idx = self.filtered_onsets.index(onset)
                    next_idx = idx + 1 if idx < len(self.filtered_onsets) - 1 else 0
                    self.goto_onset(next_idx)
                else:
                    # fallback: go to the new event in the full list
                    new_idx = self.onsets.index(onset)
                    self.goto_onset(new_idx)
            else:
                pass
        QMessageBox.information(self, "Success", f"Added {event_type} event from frame {onset} to {offset_exclusive}")
        # After successful add, reset dropdown to placeholder
        self.event_type_combo.setCurrentIndex(0)
        # Check for mouse milestones after adding manual event
        self.check_mouse_milestones()
        self.maybe_start_auto_save()
        self.maybe_stop_auto_save()
        
    def delete_current_onset(self):
        """Delete the currently selected onset"""
        if not self.onsets:
            return
            
        current_onset = self.onsets[self.current_onset_idx]
        
        # Remove from timeline
        self.timeline_canvas.remove_event(current_onset)
        
        # Remove from data structures
        self.onsets.remove(current_onset)
        if current_onset in self.onset_types:
            del self.onset_types[current_onset]
            
        # Remove from curated events
        for event_type, events in self.curated_events.items():
            self.curated_events[event_type] = [event for event in events if event[0] != current_onset]
            
        # Update current index
        if self.onsets:
            if self.current_onset_idx >= len(self.onsets):
                self.current_onset_idx = len(self.onsets) - 1
            self.goto_onset(self.current_onset_idx)
        else:
            self.current_onset_idx = 0
            self.update_onset_info()
            
    def play(self):
        if self.cap is not None:
            # Only play if FPS is set
            if hasattr(self, 'fps') and self.fps >= 1:
                interval = int(1000 / self.fps)
                # Exit loop mode when using normal play
                self.is_looping_event = False
                self.timer.start(interval)
            else:
                QMessageBox.warning(self, "Warning", "Please enter a valid FPS before playing.")
            
    def pause(self):
        if self.timer.isActive():
            self.timer.stop()
        # Exit loop mode on pause
        self.is_looping_event = False
        
    def stop(self):
        self.timer.stop()
        self.current_frame = 0
        self.frame_slider.setValue(0)
        self.show_frame(0)
        # Exit loop mode on stop
        self.is_looping_event = False
        
    def handle_fps_change(self):
        text = self.fps_lineedit.text()
        if text.isdigit() and int(text) >= 1:
            self.fps = int(text)
            # Update timer interval if playing
            if hasattr(self, 'timer') and self.timer.isActive():
                interval = int(1000 / self.fps)
                self.timer.start(interval)
        # If invalid, do not update self.fps
        # No special handling needed for loop mode; it uses the same timer

    def start_loop_playback(self):
        # Only start if current frame is exactly an onset
        if self.current_frame not in getattr(self, 'onsets', []):
            return
        # Determine corresponding offset
        onset = self.current_frame
        offset = self.timeline_canvas.event_offsets.get(onset, onset)
        # Define loop bounds with ±10 frames
        start = max(0, onset - 10)
        end = min(self.total_frames - 1, (offset + 10))
        self.loop_start_frame = start
        self.loop_end_frame = end
        self.is_looping_event = True
        # Jump to start immediately
        self.current_frame = start
        self.frame_slider.setValue(self.current_frame)
        self.show_frame(self.current_frame)
        # Start timer with chosen FPS
        if hasattr(self, 'fps') and self.fps >= 1:
            interval = int(1000 / self.fps)
            self.timer.start(interval)
        else:
            QMessageBox.warning(self, "Warning", "Please enter a valid FPS before playing.")
            
    def next_frame(self):
        if self.is_looping_event and self.loop_end_frame is not None:
            # If we've reached the loop end, jump back to loop start
            if self.current_frame >= self.loop_end_frame:
                self.current_frame = max(0, self.loop_start_frame)
                self.frame_slider.setValue(self.current_frame)
                self.show_frame(self.current_frame)
                return
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
            self.show_frame(self.current_frame)
        else:
            self.pause()
            
    def show_frame(self, frame_num):
        if self.cap is None:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Check if current frame is an onset (for status bar update)
        onset_type = self.onset_types.get(frame_num, None)
        
        # Optimize video processing and apply zoom by cropping, not resizing the panel
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        label_size = self.video_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            if self.video_zoom_factor_cam1 > 1.0:
                crop_w = max(1, int(w / self.video_zoom_factor_cam1))
                crop_h = max(1, int(h / self.video_zoom_factor_cam1))
                cx, cy = w // 2, h // 2
                x0 = max(0, cx - crop_w // 2)
                y0 = max(0, cy - crop_h // 2)
                x1 = min(w, x0 + crop_w)
                y1 = min(h, y0 + crop_h)
                cropped = frame[y0:y1, x0:x1]
            else:
                cropped = frame
            cropped = np.ascontiguousarray(cropped)
            ch2 = cropped.shape[2]
            bytes_per_line2 = ch2 * cropped.shape[1]
            q_img = QImage(cropped.tobytes(), cropped.shape[1], cropped.shape[0], bytes_per_line2, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

        # Render second camera if available
        if hasattr(self, 'cap2') and self.cap2 is not None and self.second_video_label.isVisible():
            try:
                self.cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret2, frame2 = self.cap2.read()
                if ret2:
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    h2, w2, ch2 = frame2.shape
                    label2_size = self.second_video_label.size()
                    if label2_size.width() > 0 and label2_size.height() > 0:
                        if self.video_zoom_factor_cam2 > 1.0:
                            crop_w2 = max(1, int(w2 / self.video_zoom_factor_cam2))
                            crop_h2 = max(1, int(h2 / self.video_zoom_factor_cam2))
                            cx2, cy2 = w2 // 2, h2 // 2
                            x0_2 = max(0, cx2 - crop_w2 // 2)
                            y0_2 = max(0, cy2 - crop_h2 // 2)
                            x1_2 = min(w2, x0_2 + crop_w2)
                            y1_2 = min(h2, y0_2 + crop_h2)
                            cropped2 = frame2[y0_2:y1_2, x0_2:x1_2]
                        else:
                            cropped2 = frame2
                        cropped2 = np.ascontiguousarray(cropped2)
                        ch2b = cropped2.shape[2]
                        bytes_per_line2b = ch2b * cropped2.shape[1]
                        q_img2 = QImage(cropped2.tobytes(), cropped2.shape[1], cropped2.shape[0], bytes_per_line2b, QImage.Format_RGB888)
                        pixmap2 = QPixmap.fromImage(q_img2)
                        scaled_pixmap2 = pixmap2.scaled(label2_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.second_video_label.setPixmap(scaled_pixmap2)
            except Exception:
                pass
        
        self.update_frame_info()
        self.update_onset_status()
        # Enable loop button only when at an onset frame
        try:
            if hasattr(self, 'loop_btn'):
                self.loop_btn.setEnabled(frame_num in self.onsets)
        except Exception:
            pass
        self.setFocus()
        
    def slider_moved(self, value):
        self.current_frame = value
        self.show_frame(value)
        # Sync both timelines' cursor when slider moves
        self.timeline_canvas.current_frame = value
        self.timeline_canvas.update_timeline()
        if hasattr(self, 'timeline_canvas2') and self.timeline_canvas2.isVisible():
            self.timeline_canvas2.current_frame = value
            self.timeline_canvas2.update_timeline()
        self.update_onset_info()  # Ne change pas l'onset courant
        # Toggle loop button availability
        if hasattr(self, 'loop_btn'):
            self.loop_btn.setEnabled(value in getattr(self, 'onsets', []))

    def _sync_frame_all(self, frame):
        self.current_frame = frame
        self.frame_slider.setValue(frame)
        self.show_frame(frame)
        # Update both timelines' red cursor
        self.timeline_canvas.current_frame = frame
        self.timeline_canvas.update_timeline()
        if hasattr(self, 'timeline_canvas2') and self.timeline_canvas2.isVisible():
            self.timeline_canvas2.current_frame = frame
            self.timeline_canvas2.update_timeline()
        # Only update spinbox if it's not currently being edited
        if not self.frame_spinbox.hasFocus():
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setValue(frame)
            self.frame_spinbox.blockSignals(False)
        self.update_onset_info()
        if hasattr(self, 'loop_btn'):
            self.loop_btn.setEnabled(frame in getattr(self, 'onsets', []))

    def timeline_frame_changed_primary(self, frame):
        # Set primary as active when interacting with first timeline
        self.active_timeline_index = 1
        self._sync_frame_all(frame)
        # Keep axes aligned when interacting with primary
        if hasattr(self, 'timeline_canvas2') and self.timeline_canvas2.isVisible():
            try:
                xlim = self.timeline_canvas.ax.get_xlim()
                self.timeline_canvas2.ax.set_xlim(xlim)
            except Exception:
                pass

    def timeline_frame_changed_secondary(self, frame):
        # Set secondary as active when interacting with second timeline
        self.active_timeline_index = 2
        self._sync_frame_all(frame)
        # Keep both timelines horizontally aligned when moving on secondary
        try:
            xlim = self.timeline_canvas.ax.get_xlim()
            self.timeline_canvas2.ax.set_xlim(xlim)
        except Exception:
            pass

    def update_frame_info(self):
        # Update spinbox and total frames label
        # Only update spinbox if it's not currently being edited
        if not self.frame_spinbox.hasFocus():
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setValue(self.current_frame)
            self.frame_spinbox.blockSignals(False)
        self.frame_spinbox.setMaximum(max(0, self.total_frames - 1))
        self.total_frames_label.setText(str(self.total_frames))
        
    def update_onset_filter(self):
        # Update filtered_onsets based on both filter selections
        type_text = self.onset_filter_combo.currentText().lower()
        status_text = self.status_filter_combo.currentText().lower()
        # Filter by type
        if type_text == "all":
            filtered = self.onsets.copy()
        else:
            filtered = [o for o in self.onsets if self.onset_types.get(o, '').lower() == type_text]
        # Filter by status
        if status_text != "all":
            # Handle the "Manually Added" case specifically
            if status_text == "manually added":
                filtered = [o for o in filtered if self.timeline_canvas.onset_validations.get(o, 'pending') == 'manually added']
            else:
                filtered = [o for o in filtered if self.timeline_canvas.onset_validations.get(o, 'pending').lower() == status_text]
        self.filtered_onsets = filtered
        # Si la liste filtrée est vide, ne pas fallback, afficher 0/0 et désactiver navigation
        if not self.filtered_onsets:
            self.current_onset_idx = 0
            self.update_onset_info()
            return
        # Try to keep the current frame in view if possible
        current_frame = self.current_frame
        idx = 0
        for i, onset in enumerate(self.filtered_onsets):
            if onset <= current_frame:
                idx = i
            else:
                break
        self.current_onset_idx = idx
        self.update_onset_info()
        self.check_all_validated_and_show_firework()
        
    def goto_onset(self, idx):
        # Choisir la timeline active
        if getattr(self, "active_timeline_index", 1) == 1:
            if not hasattr(self, 'filtered_onsets') or not self.filtered_onsets:
                self.filtered_onsets = self.onsets.copy()
            onsets_list = getattr(self, 'filtered_onsets', getattr(self, 'onsets', []))
            timeline = self.timeline_canvas
        else:
            if not hasattr(self, 'filtered_onsets2') or not self.filtered_onsets2:
                self.filtered_onsets2 = self.onsets2.copy()
            onsets_list = getattr(self, 'filtered_onsets2', getattr(self, 'onsets2', []))
            timeline = self.timeline_canvas2
        # Use filtered_onsets for navigation
        if 0 <= idx < len(onsets_list):
            self.current_onset_idx = idx
            onset_frame = onsets_list[idx]
            self.current_frame = onset_frame
            self.frame_slider.setValue(onset_frame)
            self.show_frame(onset_frame)
            timeline.current_frame = onset_frame
            timeline.update_timeline()
            # Only update spinbox if it's not currently being edited
            if not self.frame_spinbox.hasFocus():
                self.frame_spinbox.blockSignals(True)
                self.frame_spinbox.setValue(onset_frame)
                self.frame_spinbox.blockSignals(False)
            self.update_onset_info()
            # Toggle loop button availability
            if hasattr(self, 'loop_btn'):
                self.loop_btn.setEnabled(onset_frame in getattr(self, 'onsets', []))
            
    def prev_onset(self):
        if getattr(self, "active_timeline_index", 1) == 1:
            if not hasattr(self, 'filtered_onsets') or not self.filtered_onsets:
                self.filtered_onsets = self.onsets.copy()
            onsets_list = getattr(self, 'filtered_onsets', getattr(self, 'onsets', []))
        else:
            if not hasattr(self, 'filtered_onsets2') or not self.filtered_onsets2:
                self.filtered_onsets2 = self.onsets2.copy()
            onsets_list = getattr(self, 'filtered_onsets2', getattr(self, 'onsets2', []))

        if not onsets_list:
            return

        # Si on est au premier onset, aller au dernier (cyclique)
        if self.current_onset_idx == 0:
            idx = len(onsets_list) - 1
        else:
            idx = self.current_onset_idx - 1
        self.goto_onset(idx)
        
    def setup_offset_validation_widget(self):
        self.offset_validation_widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Validate Offset:"))
        self.offset_edit_spinbox = QSpinBox()
        self.offset_edit_spinbox.setRange(0, 999999)
        layout.addWidget(self.offset_edit_spinbox)
        self.confirm_offset_btn = QPushButton("Validate")
        self.confirm_offset_btn.clicked.connect(self.confirm_and_accept_offset)
        layout.addWidget(self.confirm_offset_btn)
        self.offset_validation_widget.setLayout(layout)
        self.offset_validation_widget.hide()

    def confirm_and_accept_offset(self):
        self.unsaved_changes = True
        new_offset_exclusive = self.offset_edit_spinbox.value()
        new_offset = new_offset_exclusive - 1  # Convert to inclusive

        if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
            current_onset = self.filtered_onsets[self.current_onset_idx]
        else:
            current_onset = self.onsets[self.current_onset_idx]

        if new_offset < current_onset:
            QMessageBox.warning(self, "Invalid Offset", "Offset must be greater than or equal to the onset.")
            return

        # Update data structures
        self.timeline_canvas.event_offsets[current_onset] = new_offset
        for event_type, events in self.curated_events.items():
            for event in events:
                if event[0] == current_onset:
                    event[1] = new_offset

        # Set validation status to 'accepted'
        prev_status = self.timeline_canvas.onset_validations.get(current_onset, 'pending')
        if prev_status != 'accepted':
            self.undo_stack.append((current_onset, prev_status))
            self.timeline_canvas.set_onset_validation(current_onset, 'accepted')

        # Hide widget and re-enable buttons
        self.offset_validation_widget.hide()
        self.accept_btn.setEnabled(True)
        self.edit_btn.setEnabled(True)
        self.reject_btn.setEnabled(True)
        self.change_type_btn.setEnabled(True)
        self.awaiting_offset_validation = False

        # Redraw and move on
        self.update_performance_display()
        self.check_mouse_milestones()
        self.check_all_validated_and_show_firework()
        self.update_onset_filter()

        # Only advance if the list is not empty
        if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
            if self.current_onset_idx < len(self.filtered_onsets) - 1:
                self.next_onset()
            else: # if it was the last one, refresh info but don't loop
                self.goto_onset(self.current_onset_idx)

        self.setFocus()

    def start_offset_validation(self):
        if not self.onsets:
            return

        # If widget is already visible, it means user wants to cancel.
        if self.offset_validation_widget.isVisible():
            self.offset_validation_widget.hide()
            self.accept_btn.setEnabled(True)
            self.edit_btn.setEnabled(True)
            self.reject_btn.setEnabled(True)
            self.change_type_btn.setEnabled(True)
            self.setFocus()
            return

        if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
            current_onset = self.filtered_onsets[self.current_onset_idx]
        else:
            current_onset = self.onsets[self.current_onset_idx]

        current_offset = self.timeline_canvas.event_offsets.get(current_onset, current_onset)
        
        # Prefill the spinbox with the current offset (exclusive, for user display)
        self.offset_edit_spinbox.setValue(current_offset + 1)
        self.offset_edit_spinbox.setMinimum(current_onset + 1) # Ensure offset is always > onset

        # Show the validation widget and disable other buttons
        self.offset_validation_widget.show()
        self.accept_btn.setEnabled(True) # Keep accept enabled to act as a cancel button
        self.edit_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)
        self.change_type_btn.setEnabled(False)
        self.setFocus()

    def next_onset(self):
        if getattr(self, "active_timeline_index", 1) == 1:
            if not hasattr(self, 'filtered_onsets') or not self.filtered_onsets:
                self.filtered_onsets = self.onsets.copy()
            onsets_list = getattr(self, 'filtered_onsets', getattr(self, 'onsets', []))
        else:
            if not hasattr(self, 'filtered_onsets2') or not self.filtered_onsets2:
                self.filtered_onsets2 = self.onsets2.copy()
            onsets_list = getattr(self, 'filtered_onsets2', getattr(self, 'onsets2', []))

        if not onsets_list:
            return
        # Aller à l'onset le plus proche strictement après la frame courante
        current_frame = self.current_frame
        next_candidates = [o for o in onsets_list if o > current_frame]
        if next_candidates:
            idx = onsets_list.index(next_candidates[0])
        else:
            # Si aucun après, cyclique : aller au premier onset
            idx = 0
        self.goto_onset(idx)

    def update_onset_info(self):
        
        if self.active_timeline_index == 1:
            onsets_list = getattr(self, 'filtered_onsets', getattr(self, 'onsets', []))
            timeline = self.timeline_canvas
        else:
            onsets_list = getattr(self, 'filtered_onsets2', getattr(self, 'onsets2', []))
            timeline = self.timeline_canvas2
            
        if not onsets_list:
            self.onset_info_label.setText("0/0\nNo onsets loaded")
            self.prev_onset_btn.setEnabled(False)
            self.next_onset_btn.setEnabled(False)
            self.accept_btn.setEnabled(False)
            self.reject_btn.setEnabled(False)
            self.edit_btn.setEnabled(False)
            return
        
         # S'assurer que current_onset_idx est dans les limites
        if not hasattr(self, 'current_onset_idx') or self.current_onset_idx >= len(onsets_list):
            self.current_onset_idx = len(onsets_list) - 1
            
        # Réactive navigation si on a des onsets filtrés
        self.prev_onset_btn.setEnabled(True)
        self.next_onset_btn.setEnabled(True)
        # Disable buttons if offset validation is active
        if not self.offset_validation_widget.isVisible():
            self.accept_btn.setEnabled(True)
            self.reject_btn.setEnabled(True)
            self.edit_btn.setEnabled(True)
        current_onset = onsets_list[self.current_onset_idx]
        onset_type = self.onset_types.get(current_onset, '')
        if onset_type not in ('active', 'twitch'):
            onset_type = ''
        validation = timeline.onset_validations.get(current_onset, 'pending')
        offset = timeline.event_offsets.get(current_onset, current_onset)
        # For display: show offset+1 if offset > onset
        display_offset = offset + 1 if offset > current_onset else offset
        info_text = f"Onset {self.current_onset_idx + 1}/{len(onsets_list)}: Frame {current_onset}"
        if offset != current_onset:
            info_text += f" to {display_offset}"
        if onset_type:
            info_text += f" | Type: {onset_type}"
        info_text += f" | Status: {validation}"
        self.onset_info_label.setText(info_text)
        
    def validate_onset(self, validation):
        self.unsaved_changes = True
        if not self.onsets:
            return
        # Always use the current filtered onset if available
        if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
            current_onset = self.filtered_onsets[self.current_onset_idx]
        else:
            current_onset = self.onsets[self.current_onset_idx]
        prev_status = self.timeline_canvas.onset_validations.get(current_onset, 'pending')
        if prev_status == validation:
            return
        # If accepting, check for total overlap and prompt BEFORE accepting, exclude itself
        if validation == 'accepted':
            onset = current_onset
            offset = self.timeline_canvas.event_offsets.get(onset, onset)
            self.check_total_overlap_and_prompt(onset, offset, exclude_onsets=onset)
        self.undo_stack.append((current_onset, prev_status))
        self.timeline_canvas.set_onset_validation(current_onset, validation)
        if validation == 'accepted':
            self.performance_metrics['true_positives'] += 1
            offset = self.timeline_canvas.event_offsets.get(current_onset, None)
            if offset is not None and offset != current_onset:
                self.awaiting_offset_validation = True
                self.current_offset_for_validation = offset
                self.goto_offset_for_validation(offset)
                self.update_onset_filter()  # Immediate update
                self.setFocus()
                return
        elif validation == 'edited':
            self.performance_metrics['true_positives'] = int(self.performance_metrics.get('true_positives', 0)) + 1
        elif validation == 'rejected':
            self.performance_metrics['false_positives'] = int(self.performance_metrics.get('false_positives', 0)) + 1
            # No navigation here; let the final next_onset() at the end handle it
        self.update_performance_display()
        self.check_mouse_milestones()
        self.check_all_validated_and_show_firework()
        self.update_onset_filter()
        # If the filtered list is empty after rejection, do not advance
        if hasattr(self, 'filtered_onsets') and not self.filtered_onsets:
            self.onset_info_label.setText("0/0\nNo onsets loaded")
            self.prev_onset_btn.setEnabled(False)
            self.next_onset_btn.setEnabled(False)
            self.accept_btn.setEnabled(False)
            self.reject_btn.setEnabled(False)
            self.edit_btn.setEnabled(False)
            self.setFocus()
            return
        if self.current_onset_idx < len(self.onsets) - 1:
            self.next_onset()
        self.setFocus()  # Restore focus to main window

    def start_edit_onset(self):
        # Prevent editing if offset validation is in progress
        if self.offset_validation_widget.isVisible():
            return

        # Si le widget d'édition est déjà visible, le fermer
        if self.edit_widget.isVisible():
            self.edit_widget.hide()
            self.accept_btn.setEnabled(True)
            self.edit_btn.setEnabled(True)
            self.reject_btn.setEnabled(True)
            self.setFocus()
            return
        # Sinon, comportement normal : ouvrir et préremplir
        if not self.onsets:
            self.setFocus()
            return
        if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
            current_onset = self.filtered_onsets[self.current_onset_idx]
        else:
            current_onset = self.onsets[self.current_onset_idx]
        offset = self.timeline_canvas.event_offsets.get(current_onset, current_onset)
        self.edit_onset_spinbox.setValue(current_onset)
        self.edit_offset_spinbox.setValue(offset + 1)  # Show exclusive offset
        self.edit_widget.show()
        self.accept_btn.setEnabled(False)
        self.edit_btn.setEnabled(True)
        self.reject_btn.setEnabled(False)
        self.update_onset_status()  # Update status to show we're in edit mode
        self.setFocus()
        self.reject_btn.setEnabled(False)
        self.setFocus()

    def finish_edit_onset(self):
        self.unsaved_changes = True
        # Update only the onset frame, not the event type
        if not self.onsets:
            return
        new_onset = self.edit_onset_spinbox.value()
        new_offset_exclusive = self.edit_offset_spinbox.value()
        new_offset = new_offset_exclusive - 1 # Convert to inclusive for internal storage
        if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
            old_onset = self.filtered_onsets[self.current_onset_idx]
        else:
            old_onset = self.onsets[self.current_onset_idx]
        # Check for total overlap and prompt BEFORE editing, exclude both old and new onset
        self.check_total_overlap_and_prompt(new_onset, new_offset, exclude_onsets=[old_onset, new_onset])
        if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
            old_onset = self.filtered_onsets[self.current_onset_idx]
            if old_onset in self.onsets:
                idx_full = self.onsets.index(old_onset)
                self.onsets[idx_full] = new_onset
            self.filtered_onsets[self.current_onset_idx] = new_onset
        else:
            old_onset = self.onsets[self.current_onset_idx]
            self.onsets[self.current_onset_idx] = new_onset
        event_type = self.onset_types[old_onset]
        # --- Capture the old offset before any changes ---
        old_offset_before_edit = self.timeline_canvas.event_offsets.get(old_onset, old_onset)
        # Si seul l'offset change (onset identique)
        if new_onset == old_onset and new_offset != old_offset_before_edit:
            reply = QMessageBox.question(self, "Accept instead?", "Wouldn't you rather accept instead?\nIf you click Accept, the event will be validated as accepted with the new offset.", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                # Appliquer la validation 'accepted' avec le nouvel offset
                self.timeline_canvas.event_offsets[old_onset] = new_offset
                for event_type, events in self.curated_events.items():
                    for event in events:
                        if event[0] == old_onset:
                            event[1] = new_offset
                prev_status = self.timeline_canvas.onset_validations.get(old_onset, 'pending')
                if prev_status != 'accepted':
                    self.undo_stack.append((old_onset, prev_status))
                self.timeline_canvas.onset_validations[old_onset] = 'accepted'
                self.edit_widget.hide()
                self.accept_btn.setEnabled(True)
                self.edit_btn.setEnabled(True)
                self.reject_btn.setEnabled(True)
                self.update_onset_status()
                self.update_onset_filter()
                self.update_onset_info()
                self.update_performance_display()
                self.timeline_canvas.plot_motion_energy_preserve_view(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, getattr(self, 'event_status', None))
                # Aller à l'onset suivant
                if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
                    if self.current_onset_idx < len(self.filtered_onsets) - 1:
                        self.goto_onset(self.current_onset_idx + 1)
                else:
                    if self.current_onset_idx < len(self.onsets) - 1:
                        self.goto_onset(self.current_onset_idx + 1)
                self.setFocus()
                return
            # Sinon, continuer l'édition normale (statut 'edited')
        del self.onset_types[old_onset]
        self.onset_types[new_onset] = event_type
        self.timeline_canvas.event_offsets[new_onset] = new_offset  # Save as inclusive last frame
        if old_onset in self.timeline_canvas.event_offsets:
            del self.timeline_canvas.event_offsets[old_onset]
        for events in self.curated_events.values():
            for event in events:
                if event[0] == old_onset:
                    event[0] = new_onset
                    event[1] = new_offset
        # Stocke l'historique complet des anciens onsets/offsets pour l'export MF
        if not hasattr(self, 'edited_onsets'):
            self.edited_onsets = {}
        if new_onset not in self.edited_onsets:
            self.edited_onsets[new_onset] = []
        # Ajoute l'ancien onset/offset à la liste
        self.edited_onsets[new_onset].append((old_onset, old_offset_before_edit))
        # Find the original onset for this event
        orig_onset = self.original_onsets.get(old_onset, old_onset)
        shift = abs(new_onset - orig_onset)
        # Update original_onsets mapping
        self.original_onsets[new_onset] = orig_onset
        if old_onset in self.original_onsets:
            del self.original_onsets[old_onset]
        # Set validation and score based on shift
        prev_status = self.timeline_canvas.onset_validations.get(new_onset, 'pending')
        if prev_status != 'edited':
            self.undo_stack.append((new_onset, prev_status, old_onset, orig_onset, old_offset_before_edit, self.original_offsets.get(old_onset, old_offset_before_edit)))
        self.timeline_canvas.onset_validations[new_onset] = 'edited'
        if hasattr(self, 'event_status'):
            if shift < self.edit_threshold:
                self.event_status[new_onset] = 1
            else:
                self.event_status[new_onset] = 0.5
            if old_onset in self.event_status:
                del self.event_status[old_onset]
        self.edit_widget.hide()
        self.accept_btn.setEnabled(True)
        self.edit_btn.setEnabled(True)
        self.reject_btn.setEnabled(True)
        self.update_onset_status()  # Update status when exiting edit mode
        self.onsets = sorted(self.onsets)
        if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
            self.filtered_onsets = sorted(self.filtered_onsets)
            self.current_onset_idx = self.filtered_onsets.index(new_onset)
        else:
            self.current_onset_idx = self.onsets.index(new_onset)
        self.timeline_canvas.plot_motion_energy_preserve_view(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, getattr(self, 'event_status', None))
        self.update_onset_info()
        self.update_performance_display()
        # Aller à l'onset suivant après édition
        if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
            if self.current_onset_idx < len(self.filtered_onsets) - 1:
                self.goto_onset(self.current_onset_idx + 1)
        else:
            if self.current_onset_idx < len(self.onsets) - 1:
                self.goto_onset(self.current_onset_idx + 1)
        # Mise à jour du filtre et du compteur après édition
        self.update_onset_filter()
        self.setFocus()  # Restore focus to main window
        # Check for total overlap and prompt
        self.check_total_overlap_and_prompt(new_onset, new_offset, exclude_onsets=[old_onset, new_onset])

    def update_performance_display(self):
        # New: Score is 1 for accepted, -1 for rejected, 0.5 for edited, 0 for pending/manually added
        scores = []
        for onset in self.onsets:
            status = self.timeline_canvas.onset_validations.get(onset, 'pending')
            if status == 'accepted':
                scores.append(1)
            elif status == 'rejected':
                scores.append(-1)
            elif status == 'edited':
                scores.append(0.5)
            elif status == 'manually added':
                scores.append(0)
            else:
                scores.append(0)
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 0
        display_text = f"""
Average Score: {avg_score:.3f}
(accepted=1, rejected=-1, edited=0.5, pending=0, manually added=0)
        """
        self.metrics_text.setText(display_text.strip())
        self.check_mouse_milestones()
        self.check_all_validated_and_show_firework()
        
    # Removed save_curated_onsets() - functionality merged into save_and_export_validation()
            
    def save_and_export_validation(self):
        """Combined function to save validation data, performance metrics, and create comparison plot"""
        if not self.curated_events:
            QMessageBox.warning(self, "Warning", "No onsets to save")
            return
        overlaps = self.check_for_overlaps()
        if overlaps:
            overlap_str = "\n".join([f"{a1}-{b1} overlaps {a2}-{b2}" for a1, b1, a2, b2 in overlaps])
            reply = QMessageBox.question(
                self,
                "Overlapping Events",
                f"The following events overlap:\n\n{overlap_str}\n\nDo you want to save/export anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        # Create a flat list of events with 5 columns each
        export_events = []
        for event_type, events in self.curated_events.items():
            for event in events:
                onset = event[0]
                offset = event[1] if len(event) > 1 else onset
                # For export: make offset inclusive
                export_offset = offset + 1 if offset > onset else offset
                status = self.timeline_canvas.onset_validations.get(onset, 'pending')
                if status == 'accepted':
                    score = 1
                elif status == 'rejected':
                    score = -1
                elif status == 'edited':
                    score = 0.5
                elif status == 'manually added':
                    score = 0
                else:
                    score = 0
                export_events.append([onset, export_offset, event_type, status, score])

        # Save to XLSX file only
        fname, _ = QFileDialog.getSaveFileName(self, 'Save and Export Validation', '', 'Excel (*.xlsx)')
        if fname:
            if not fname.endswith('.xlsx'):
                fname += '.xlsx'
            import os
            import pandas as pd
            base_dir = os.path.dirname(fname)
            output_dir = os.path.join(base_dir, "mousecraft_output")
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(fname))[0]
            # Save XLSX in output directory
            xlsx_path = os.path.join(output_dir, f"{base_name}.xlsx")
            df = pd.DataFrame(export_events, columns=pd.Index(['onset', 'offset', 'event_type', 'status', 'score']))
            df.to_excel(xlsx_path, index=False)
            # Save numpy file in output directory
            np_path = os.path.join(output_dir, f"{base_name}.npy")
            np.save(np_path, np.array(export_events, dtype=object))
            # Save performance metrics as JSON in output directory
            metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
            metrics = self.performance_metrics
            total = metrics['true_positives'] + metrics['false_positives']
            if total > 0:
                precision = metrics['true_positives'] / total if total > 0 else 0
            else:
                precision = 0
            results = {
                'performance_metrics': metrics,
                'precision': precision,
                'total_annotated': total
            }
            with open(metrics_path, 'w') as f:
                json.dump(results, f, indent=2)
            # Create comparison plot and save it in output directory
            plot_path = os.path.join(output_dir, f"{base_name}_comparison_plot.png")
            self.create_validation_comparison_plot(export_events, save_path=plot_path)
            # Show success message with all saved files
            QMessageBox.information(self, "Success", 
                f"Validation exported successfully!\n\n"
                f"Files saved in 'mousecraft_output' directory:\n"
                f"• {base_name}.xlsx (Excel data)\n"
                f"• {base_name}.npy (NumPy data)\n"
                f"• {base_name}_metrics.json (Performance metrics)\n"
                f"• {base_name}_comparison_plot.png (Comparison plot)\n\n"
                f"Directory: {output_dir}")
            
        # Après export, reset le flag
        self.unsaved_changes = False
            
    def create_validation_comparison_plot(self, export_events, save_path=None):
        """Create a comparison plot showing original vs validated motion energy classification"""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        
        # Calculate statistics
        accepted_count = sum(1 for event in export_events if event[3] == 'accepted')
        edited_count = sum(1 for event in export_events if event[3] == 'edited')
        rejected_count = sum(1 for event in export_events if event[3] == 'rejected')
        manually_added_count = sum(1 for event in export_events if event[3] == 'manually added')
        pending_count = sum(1 for event in export_events if event[3] == 'pending')
        
        total_events = len(export_events)
        true_positives = accepted_count + edited_count  # Accepted + Edited
        false_positives = rejected_count  # Rejected
        
        # Check if motion_energy exists
        if self.motion_energy is None:
            QMessageBox.warning(self, "Warning", "No motion energy data available for comparison plot")
            return
            
        # Create the comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5.5), gridspec_kw={'height_ratios': [0.38, 0.38]})
        plt.subplots_adjust(top=0.85)
        
        # Top subplot: Original classification (all events as accepted)
        ax1.plot(np.arange(len(self.motion_energy)), self.motion_energy, color='blue', linewidth=1, alpha=0.7)
        
        # Plot all original events as accepted (purple for twitch, yellow for active)
        for event in export_events:
            onset, offset, event_type, status, score = event
            if event_type == 'twitch':
                ax1.axvspan(onset, offset, color='purple', alpha=0.6)
            elif event_type == 'active':
                ax1.axvspan(onset, offset, color='yellow', alpha=0.6)
        
        ax1.set_title('Original Motion Energy Classification (All Events)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Motion Energy', fontsize=12)
        ax1.set_xlim(0, len(self.motion_energy))
        ax1.set_ylim(0, max(self.motion_energy) * 1.1)
        ax1.grid(True, alpha=0.3)
        try:
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
        except Exception:
            pass
        
        # Bottom subplot: Validated classification (with validation markers)
        ax2.plot(np.arange(len(self.motion_energy)), self.motion_energy, color='blue', linewidth=1, alpha=0.7)
        
        # Plot events with validation status
        for event in export_events:
            onset, offset, event_type, status, score = event
            
            # Set alpha based on validation status
            if status == 'accepted':
                alpha = 1.0
            elif status == 'edited':
                alpha = 0.8
            elif status == 'rejected':
                alpha = 0.3
            elif status == 'manually added':
                alpha = 0.5
            else:  # pending
                alpha = 0.4
                
            if event_type == 'twitch':
                ax2.axvspan(onset, offset, color='purple', alpha=alpha)
            elif event_type == 'active':
                ax2.axvspan(onset, offset, color='yellow', alpha=alpha)
        
        # Add validation markers with same logic as timeline
        for event in export_events:
            onset, offset, event_type, status, score = event
            if status == 'accepted':
                ax2.plot(onset, max(self.motion_energy) * 1.05, 'o', color='green', markersize=6)
            elif status == 'rejected':
                ax2.plot(onset, max(self.motion_energy) * 1.05, 'o', color='red', markersize=6)
            elif status == 'edited':
                # Use same logic as timeline for edited events
                if hasattr(self, 'event_status') and onset in self.event_status:
                    score = self.event_status[onset]
                else:
                    score = 0.5  # Default for edited events
                
                if score == 1:
                    color = 'green'
                elif score == 0.5:
                    color = 'orange'
                else:
                    color = 'orange'  # Fallback
                ax2.plot(onset, max(self.motion_energy) * 1.05, 'o', color=color, markersize=6)
            elif status == 'manually added':
                ax2.plot(onset, max(self.motion_energy) * 1.05, 'o', color='blue', markersize=6)
        
        ax2.set_title('Validated Motion Energy Classification', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Frame', fontsize=12)
        ax2.set_ylabel('Motion Energy', fontsize=12)
        ax2.set_xlim(0, len(self.motion_energy))
        ax2.set_ylim(0, max(self.motion_energy) * 1.2)
        ax2.grid(True, alpha=0.3)
        try:
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
        except Exception:
            pass
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], color='purple', alpha=0.6, linewidth=10, label='Twitch'),
            Line2D([0], [0], color='yellow', alpha=0.6, linewidth=10, label='Active'),
            Line2D([0], [0], marker='o', color='green', markersize=8, label='Accepted', linestyle=''),
            Line2D([0], [0], marker='o', color='red', markersize=8, label='Rejected', linestyle=''),
            Line2D([0], [0], marker='o', color='orange', markersize=8, label='Edited (orange)', linestyle=''),
            Line2D([0], [0], marker='o', color='green', markersize=8, label='Edited (green)', linestyle=''),
            Line2D([0], [0], marker='o', color='#00CED1', markersize=8, label='Manually Added', linestyle='')
        ]
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.01, 0.99), loc='upper left', ncol=1, fontsize=11, borderaxespad=0.)
        
        plt.tight_layout()
        
        # Show statistics in a text box
        stats_text = (
            f"Validation: Total Events: {total_events}, "
            f"Accepted: {accepted_count} ({accepted_count/total_events*100:.1f}%), "
            f"Edited: {edited_count} ({edited_count/total_events*100:.1f}%), "
            f"Rejected: {rejected_count} ({rejected_count/total_events*100:.1f}%), "
            f"Manually Added: {manually_added_count} ({manually_added_count/total_events*100:.1f}%), "
            f"Pending: {pending_count} ({pending_count/total_events*100:.1f}%)\n"
            f"Performance: True Positives (Accepted+Edited): {true_positives} ({true_positives/total_events*100:.1f}%), "
            f"False Positives (Rejected): {false_positives} ({false_positives/total_events*100:.1f}%), "
            f"Precision: {true_positives/(true_positives+false_positives)*100:.1f}% (if no pending/manually added)"
        )
        
        # Add statistics text box
        plt.figtext(0.01, 0.01, stats_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8), va='bottom', ha='left', wrap=True)
        
        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
            
    # Removed save_performance_metrics() - functionality merged into save_and_export_validation()

    def load_framewise_table(self, fname):
        import pandas as pd
        df = pd.read_csv(fname) if fname.endswith('.csv') else pd.read_excel(fname)
        # Clean and fill missing status values
        if 'status' in df.columns:
            df['status'] = df['status'].fillna('pending').astype(str).str.strip().str.lower()

        # Motion energy is required for timeline scaling (keep original indexing; do NOT average here)
        if 'motion_energy' in df.columns:
            self.motion_energy = pd.to_numeric(df['motion_energy'], errors='coerce').fillna(0).values.astype(np.float64)
            # Keep indices 1:1 with dataframe rows to align onsets/offsets
            self.total_frames = len(self.motion_energy)
            self.frame_slider.setMaximum(max(0, self.total_frames - 1))
            self.onset_spinbox.setMaximum(max(0, self.total_frames - 1))
            self.offset_spinbox.setMaximum(max(0, self.total_frames - 1))
        else:
            # Fallback: if not provided, create a zero signal length = max frame_idx or infer from first binary col
            if 'frame_idx' in df.columns:
                n = int(df['frame_idx'].max()) + 1
            else:
                # Find first numeric col to infer length
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                n = len(df[num_cols[0]]) if num_cols else 1000
            self.motion_energy = np.zeros(n, dtype=np.float64)

        # Reset structures
        self.onsets = []
        self.onset_types = {}
        self.timeline_canvas.event_offsets = {}
        self.event_status = {}
        self.curated_events = {}
        self.timeline_canvas.onset_validations = {}

        # Detect any number of event-type columns (binary masks) dynamically
        # Rule: numeric columns not in ignore list AND values exclusively in {0,1} with at least one 1
        ignore_cols = {'motion_energy', 'status', 'score', 'frame_idx'}
        candidate_cols = []
        for c in df.columns:
            if c in ignore_cols:
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            try:
                series = pd.to_numeric(df[c], errors='coerce')
                vals = pd.unique(series.dropna().round(0))
                uniques = set([float(v) for v in vals.tolist()])
                if uniques.issubset({0.0, 1.0}):
                    # Must contain at least one 1 to be considered an event type
                    if (series.fillna(0).round(0).astype(int) == 1).any():
                        candidate_cols.append(c)
            except Exception:
                continue
        try:
            print(f"[Framewise Loader] Detected binary event columns: {candidate_cols}")
        except Exception:
            pass

        # Ensure dynamic palette exists
        if not hasattr(self, '_auto_palette'):
            self._auto_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            ]

        for col in candidate_cols:
            etype = str(col).lower().strip()
            # Register type and assign color if new
            if etype not in self.available_event_types:
                self.available_event_types.append(etype)
            if etype not in self.event_type_colors:
                used = {c.lower() for c in self.event_type_colors.values()}
                picked = None
                for c in self._auto_palette:
                    if c.lower() not in used:
                        picked = c
                        break
                if picked is None:
                    picked = f"#{hash(etype) & 0xFFFFFF:06x}"
                self.event_type_colors[etype] = picked

            # Parse binary mask into intervals
            arr = df[col].fillna(0).astype(int).values
            in_event = False
            onset = None
            created = 0
            for i, val in enumerate(arr):
                if val == 1 and not in_event:
                    onset = i
                    in_event = True
                elif (val == 0 or i == len(arr) - 1) and in_event:
                    offset = i - 1 if val == 0 else i
                    if onset not in self.onsets:
                        self.onsets.append(onset)
                        self.onset_types[onset] = etype
                        self.timeline_canvas.event_offsets[onset] = offset
                        created += 1
                    # Optional status/score if present (attach to onset)
                    status = str(df.at[onset, 'status']).strip().lower() if 'status' in df.columns else 'pending'
                    score = df.at[onset, 'score'] if 'score' in df.columns else 0
                    if status == 'edited':
                        gui_status = 'edited'; gui_score = 1 if float(score) == 1 else 0.5
                    elif status == 'accepted':
                        gui_status = 'accepted'; gui_score = 1
                    elif status == 'rejected':
                        gui_status = 'rejected'; gui_score = -1
                    elif status == 'manually added':
                        gui_status = 'manually added'; gui_score = 0
                    else:
                        gui_status = 'pending'; gui_score = 0
                    self.timeline_canvas.onset_validations[onset] = gui_status
                    self.event_status[onset] = gui_score
                    if etype not in self.curated_events:
                        self.curated_events[etype] = []
                    self.curated_events[etype].append([onset, offset, 1])
                    in_event = False
            # If an event is still open at the very end, close it
            if in_event and onset is not None:
                offset = len(arr) - 1
                if onset not in self.onsets:
                    self.onsets.append(onset)
                    self.onset_types[onset] = etype
                    self.timeline_canvas.event_offsets[onset] = offset
                    created += 1
                status = str(df.at[onset, 'status']).strip().lower() if 'status' in df.columns else 'pending'
                score = df.at[onset, 'score'] if 'score' in df.columns else 0
                if status == 'edited':
                    gui_status = 'edited'; gui_score = 1 if float(score) == 1 else 0.5
                elif status == 'accepted':
                    gui_status = 'accepted'; gui_score = 1
                elif status == 'rejected':
                    gui_status = 'rejected'; gui_score = -1
                elif status == 'manually added':
                    gui_status = 'manually added'; gui_score = 0
                else:
                    gui_status = 'pending'; gui_score = 0
                self.timeline_canvas.onset_validations[onset] = gui_status
                self.event_status[onset] = gui_score
                if etype not in self.curated_events:
                    self.curated_events[etype] = []
                self.curated_events[etype].append([onset, offset, 1])

        # Finalize and update UI
        self.onsets = sorted(set(self.onsets))
        # Update combos with all event types
        if hasattr(self, 'event_type_combo'):
            placeholder = self.event_type_combo.itemText(0) if self.event_type_combo.count() > 0 else "Select type…"
            self.event_type_combo.clear()
            self.event_type_combo.addItem(placeholder)
            self.event_type_combo.addItems(self.available_event_types)
            self.event_type_combo.setCurrentIndex(0)
        if hasattr(self, 'onset_filter_combo'):
            current_idx = self.onset_filter_combo.currentIndex() if self.onset_filter_combo.count() > 0 else 0
            self.onset_filter_combo.clear()
            self.onset_filter_combo.addItems(["All"] + [t.capitalize() for t in self.available_event_types])
            if current_idx < self.onset_filter_combo.count():
                self.onset_filter_combo.setCurrentIndex(current_idx)
        if hasattr(self, 'change_type_dropdown') and self.change_type_dropdown is not None:
            self.change_type_dropdown.clear()
            self.change_type_dropdown.addItems(self.available_event_types)
        # Propagate colors to canvases
        # Decide default visibility: if both active and twitch exist, hide complex by default
        default_visible = list(self.available_event_types)
        if 'active' in default_visible and 'twitch' in default_visible and 'complex' in default_visible:
            default_visible = [t for t in default_visible if t != 'complex']
        if hasattr(self, 'timeline_canvas'):
            self.timeline_canvas.event_type_colors = dict(self.event_type_colors)
            # Overwrite visible_event_types to prevent stale state from previous loads
            self.timeline_canvas.visible_event_types = list(default_visible)
        if hasattr(self, 'timeline_canvas2'):
            self.timeline_canvas2.event_type_colors = dict(self.event_type_colors)
            self.timeline_canvas2.visible_event_types = list(default_visible)

        self.current_onset_idx = 0
        self.timeline_canvas.plot_motion_energy_preserve_view(
            self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, getattr(self, 'event_status', None)
        )
        if self.onsets:
            self.goto_onset(0)
        self.undo_btn.setEnabled(True)
        # Reset mouse milestone state
        self._mouse_half_shown = False
        self._mouse_milestones = set()
        self._mouse_milestones_shown = set()

    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus()

    # Allow moving the movie with left/right arrow keys and by dragging the timeline
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            if self.current_frame < self.total_frames - 1:
                self.current_frame += 1
                self.frame_slider.setValue(self.current_frame)
                self.show_frame(self.current_frame)
                self.timeline_canvas.current_frame = self.current_frame
                self.timeline_canvas.update_timeline()
        elif event.key() == Qt.Key_Left:
            if self.current_frame > 0:
                self.current_frame -= 1
                self.frame_slider.setValue(self.current_frame)
                self.show_frame(self.current_frame)
                self.timeline_canvas.current_frame = self.current_frame
                self.timeline_canvas.update_timeline()
        elif event.key() == Qt.Key_Space:
            if self.timer.isActive():
                self.pause()
            else:
                self.play()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        # Enable zooming on the video with the trackpad or mouse wheel
        if event.type() == event.MouseButtonPress:
            print("Mouse click event on:", obj)
        if (obj == self.video_label or obj == self.second_video_label) and event.type() == event.Wheel:
            # Zoom only the targeted camera; do not alter layout sizes
            if obj == self.second_video_label:
                if event.angleDelta().y() > 0:
                    self.video_zoom_factor_cam2 = min(self.video_zoom_factor_cam2 * 1.2, 3.0)
                else:
                    self.video_zoom_factor_cam2 = max(self.video_zoom_factor_cam2 / 1.2, 0.3)
            else:
                if event.angleDelta().y() > 0:
                    self.video_zoom_factor_cam1 = min(self.video_zoom_factor_cam1 * 1.2, 3.0)
                else:
                    self.video_zoom_factor_cam1 = max(self.video_zoom_factor_cam1 / 1.2, 0.3)
            # Redraw current frame without affecting layout
            if self.cap is not None:
                self.show_frame(self.current_frame)
            return True
        if obj == self.second_video_label and event.type() == event.Show:
            try:
                self.video_stack_layout.setStretch(0, 1)
                self.video_stack_layout.setStretch(1, 1)
                self.reset_video_zoom(target=2)
            except Exception:
                pass
            return False
        return super().eventFilter(obj, event)

    def zoom_in_timeline(self):
        # Zoom in on the x-axis by a factor of 2, centered on current frame
        # Use shared extent across both inputs so both can zoom consistently
        ax1 = self.timeline_canvas.ax
        xlim = ax1.get_xlim()
        center = self.current_frame
        width = max(1, xlim[1] - xlim[0])
        new_width = width / 2
        total1 = getattr(self.timeline_canvas, 'total_frames', 0)
        total2 = getattr(self.timeline_canvas2, 'total_frames', 0) if hasattr(self, 'timeline_canvas2') and self.timeline_canvas2.isVisible() else 0
        xmax = max(total1, total2, 1000)
        new_xlim = (max(center - new_width/2, 0), min(center + new_width/2, xmax))
        # Apply to both timelines
        ax1.set_xlim(new_xlim)
        self.timeline_canvas.draw()
        self.timeline_canvas.flush_events()
        if hasattr(self, 'timeline_canvas2') and self.timeline_canvas2.isVisible():
            ax2 = self.timeline_canvas2.ax
            ax2.set_xlim(new_xlim)
            self.timeline_canvas2.draw()
        
    def zoom_out_timeline(self):
        # Zoom out on the x-axis by a factor of 2, centered on current frame
        # Use shared extent across both inputs so both can zoom consistently
        ax1 = self.timeline_canvas.ax
        xlim = ax1.get_xlim()
        center = self.current_frame
        width = max(1, xlim[1] - xlim[0])
        new_width = width * 2
        total1 = getattr(self.timeline_canvas, 'total_frames', 0)
        total2 = getattr(self.timeline_canvas2, 'total_frames', 0) if hasattr(self, 'timeline_canvas2') and self.timeline_canvas2.isVisible() else 0
        xmax = max(total1, total2, 1000)
        new_xlim = (max(center - new_width/2, 0), min(center + new_width/2, xmax))
        # Apply to both timelines
        ax1.set_xlim(new_xlim)
        self.timeline_canvas.draw()
        self.timeline_canvas.flush_events()
        if hasattr(self, 'timeline_canvas2') and self.timeline_canvas2.isVisible():
            ax2 = self.timeline_canvas2.ax
            ax2.set_xlim(new_xlim)
            self.timeline_canvas2.draw()

    def reset_zoom_timeline(self):
        # Reset both timelines to show their full extents and keep x-axes aligned
        # Compute global xmax across both inputs (fallback to 1000 if empty)
        total1 = getattr(self.timeline_canvas, 'total_frames', 0)
        total2 = getattr(self.timeline_canvas2, 'total_frames', 0) if hasattr(self, 'timeline_canvas2') and self.timeline_canvas2.isVisible() else 0
        xmax = max(total1, total2, 1000)

        # Primary timeline
        ax1 = self.timeline_canvas.ax
        ax1.set_xlim(0, xmax)
        if getattr(self, 'motion_energy', None) is not None and len(self.motion_energy) > 0:
            ax1.set_ylim(0, max(self.motion_energy) * 1.2)
        else:
            ax1.set_ylim(0, 1.2)
        self.timeline_canvas.draw()
        self.timeline_canvas.flush_events()

        # Secondary timeline
        if hasattr(self, 'timeline_canvas2') and self.timeline_canvas2.isVisible():
            ax2 = self.timeline_canvas2.ax
            ax2.set_xlim(0, xmax)
            if getattr(self.timeline_canvas2, 'motion_energy', None) is not None and len(self.timeline_canvas2.motion_energy) > 0:
                ax2.set_ylim(0, max(self.timeline_canvas2.motion_energy) * 1.2)
            else:
                ax2.set_ylim(0, 1.2)
            self.timeline_canvas2.draw()

    def zoom_in_video(self, target=1):
        """Zoom in on the specified video (1 or 2)"""
        if target == 2:
            self.video_zoom_factor_cam2 = min(self.video_zoom_factor_cam2 * 1.2, 3.0)
        else:
            self.video_zoom_factor_cam1 = min(self.video_zoom_factor_cam1 * 1.2, 3.0)
        if self.cap is not None:
            self.show_frame(self.current_frame)
            
    def zoom_out_video(self, target=1):
        """Zoom out on the specified video (1 or 2)"""
        if target == 2:
            self.video_zoom_factor_cam2 = max(self.video_zoom_factor_cam2 / 1.2, 0.3)
        else:
            self.video_zoom_factor_cam1 = max(self.video_zoom_factor_cam1 / 1.2, 0.3)
        if self.cap is not None:
            self.show_frame(self.current_frame)
            
    def reset_video_zoom(self, target=None):
        """Reset video zoom to original size"""
        if target == 2:
            self.video_zoom_factor_cam2 = 1.0
        elif target == 1:
            self.video_zoom_factor_cam1 = 1.0
        else:
            self.video_zoom_factor_cam1 = 1.0
            self.video_zoom_factor_cam2 = 1.0
        if self.cap is not None:
            self.show_frame(self.current_frame)

    def update_onset_status(self):
        """Update the dual onset status bars for input 1 (left) and input 2 (right)"""
        # Helper to compute status text and style for a given input's data
        def compute_status(for_input_2=False):
            if for_input_2:
                onset_types = getattr(self, 'onset_types2', {})
                event_offsets = getattr(self.timeline_canvas2, 'event_offsets', {}) if hasattr(self, 'timeline_canvas2') else {}
            else:
                onset_types = getattr(self, 'onset_types', {})
                event_offsets = getattr(self.timeline_canvas, 'event_offsets', {}) if hasattr(self, 'timeline_canvas') else {}

            onset_type = onset_types.get(self.current_frame, None)
            is_offset = any((offset + 1) == self.current_frame for _, offset in event_offsets.items())
            is_editing = (hasattr(self, 'edit_widget') and self.edit_widget.isVisible()) or \
                         (hasattr(self, 'offset_edit_widget') and self.offset_edit_widget.isVisible())

            if onset_type:
                text = f"Frame {self.current_frame}: {onset_type.upper()} ONSET"
                if onset_type == 'twitch':
                    style = "background-color: purple; color: white; padding: 5px; border: 1px solid gray; font-weight: bold;"
                elif onset_type == 'active':
                    style = "background-color: yellow; color: black; padding: 5px; border: 1px solid gray; font-weight: bold;"
                else:
                    style = "background-color: lightgray; padding: 5px; border: 1px solid gray;"
            elif is_offset:
                text = f"Frame {self.current_frame}: <span style='color:#ff4444;font-weight:bold'>OFFSET{' (EDITING)' if is_editing else ''}</span>"
                style = "background-color: #ffcccc; color: #ff4444; padding: 5px; border: 1px solid gray; font-weight: bold;"
            else:
                text = f"Frame {self.current_frame}: No onset"
                style = "background-color: lightgray; padding: 5px; border: 1px solid gray;"
            return text, style

        # Compute both sides
        left_text, left_style = compute_status(for_input_2=False)
        right_text, right_style = compute_status(for_input_2=True)

        # Update labels if they exist (fallback to legacy single label if not migrated)
        if hasattr(self, 'onset_status_label_left') and hasattr(self, 'onset_status_label_right'):
            self.onset_status_label_left.setText(left_text)
            self.onset_status_label_left.setStyleSheet(left_style)
            self.onset_status_label_right.setText(right_text)
            self.onset_status_label_right.setStyleSheet(right_style)
        elif hasattr(self, 'onset_status_label'):
            # Backward compatibility: show left status
            self.onset_status_label.setText(left_text)
            self.onset_status_label.setStyleSheet(left_style)
        # Keep loop button in sync whenever status updates
        if hasattr(self, 'loop_btn'):
            self.loop_btn.setEnabled(self.current_frame in getattr(self, 'onsets', []))

    def goto_offset_for_validation(self, offset):
        self.current_frame = offset
        self.frame_slider.setValue(offset)
        self.show_frame(offset)
        self.timeline_canvas.current_frame = offset
        self.timeline_canvas.update_timeline()
        self.offset_validation_widget.show()
        # Optionally, update the onset info label to indicate offset validation
        self.onset_info_label.setText(f"Validate OFFSET at frame {offset} (for last accepted onset)")
        # Ne pas désactiver les boutons ici, tous doivent rester actifs
        # self.prev_onset_btn.setEnabled(False)
        # self.next_onset_btn.setEnabled(False)
        # self.accept_btn.setEnabled(False)
        # self.reject_btn.setEnabled(False)
        # self.edit_btn.setEnabled(False)

    def validate_offset(self):
        # This method is now obsolete and can be removed.
        # The logic is handled by start_offset_validation and confirm_and_accept_offset.
        pass

    def edit_offset(self):
        # This method is now obsolete and can be removed.
        # The logic is handled by start_offset_validation and confirm_and_accept_offset.
        pass

    def confirm_edit_offset(self):
        # This method is now obsolete and can be removed.
        # The logic is handled by start_offset_validation and confirm_and_accept_offset.
        pass

    def undo_last_action(self):
        self.unsaved_changes = True
        if not self.undo_stack:
            QMessageBox.information(self, "Undo", "Nothing to undo!")
            return
        action = self.undo_stack[-1]  # Peek to show info before popping
        # Determine which event and info to show
        if isinstance(action, tuple) and len(action) == 4 and action[0] == 'add_manual':
            _, onset, event_type, offset = action
            status = self.timeline_canvas.onset_validations.get(onset, 'manually added')
        elif isinstance(action, tuple) and len(action) == 6:
            new_onset, prev_status, old_onset, orig_onset, old_offset, orig_offset = action
            onset = new_onset
            offset = self.timeline_canvas.event_offsets.get(new_onset, '?')
            status = self.timeline_canvas.onset_validations.get(new_onset, 'pending')
        elif isinstance(action, tuple) and len(action) == 2:
            onset, prev_status = action
            offset = self.timeline_canvas.event_offsets.get(onset, '?')
            status = self.timeline_canvas.onset_validations.get(onset, 'pending')
        else:
            onset = '?'
            offset = '?'
            status = '?'
        reply = QMessageBox.question(self, "Confirm Undo", f"Are you sure you want to undo onset {onset} to offset {offset} (status: {status})?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        action = self.undo_stack.pop()
        
        # Cas ajout manuel : supprimer l'événement ajouté
        if isinstance(action, tuple) and len(action) == 4 and action[0] == 'add_manual':
            _, onset, event_type, offset = action
            # Remove from timeline
            self.timeline_canvas.remove_event(onset)
            # Remove from data structures
            if onset in self.onsets:
                self.onsets.remove(onset)
            if onset in self.onset_types:
                del self.onset_types[onset]
            if onset in self.timeline_canvas.event_offsets:
                del self.timeline_canvas.event_offsets[onset]
            if onset in self.timeline_canvas.onset_validations:
                del self.timeline_canvas.onset_validations[onset]
            # Remove from curated events
            for event_type, events in self.curated_events.items():
                self.curated_events[event_type] = [event for event in events if event[0] != onset]
            # Refresh view and counter
            self.update_onset_filter()
            self.update_onset_info()
            self.redraw()
            QMessageBox.information(self, "Event deleted", "This manually added event has been successfully deleted.")
            return
            
        # Cas édition d'onset : restaurer l'onset et offset précédents
        if isinstance(action, tuple) and len(action) == 6:
            new_onset, prev_status, old_onset, orig_onset, old_offset, orig_offset = action
            # Restore old onset and offset
            if new_onset in self.onsets:
                idx = self.onsets.index(new_onset)
                self.onsets[idx] = old_onset
            # Restore event type
            event_type = self.onset_types.get(new_onset, '')
            if new_onset in self.onset_types:
                del self.onset_types[new_onset]
            self.onset_types[old_onset] = event_type
            # Restore offset
            if new_onset in self.timeline_canvas.event_offsets:
                del self.timeline_canvas.event_offsets[new_onset]
            self.timeline_canvas.event_offsets[old_onset] = old_offset
            # Restore validation status
            if new_onset in self.timeline_canvas.onset_validations:
                del self.timeline_canvas.onset_validations[new_onset]
            self.timeline_canvas.onset_validations[old_onset] = prev_status
            # Restore original mappings
            self.original_onsets[old_onset] = orig_onset
            if new_onset in self.original_onsets:
                del self.original_onsets[new_onset]
            self.original_offsets[old_onset] = orig_offset
            # Update curated events (force offset update for all matching onsets)
            for event_type, events in self.curated_events.items():
                for event in events:
                    if event[0] == new_onset or event[0] == old_onset:
                        event[0] = old_onset
                        event[1] = old_offset
            # Remove from edited_onsets
            if hasattr(self, 'edited_onsets') and new_onset in self.edited_onsets:
                del self.edited_onsets[new_onset]
            # Remove from event_status
            if hasattr(self, 'event_status') and new_onset in self.event_status:
                del self.event_status[new_onset]
            # Sort onsets
            self.onsets = sorted(self.onsets)
            # Update current index
            if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
                if old_onset in self.filtered_onsets:
                    self.current_onset_idx = self.filtered_onsets.index(old_onset)
            else:
                if old_onset in self.onsets:
                    self.current_onset_idx = self.onsets.index(old_onset)
            # Refresh view
            self.update_onset_filter()
            self.update_onset_info()
            self.redraw()
            # Show both onset and offset in the message
            new_offset_val = self.timeline_canvas.event_offsets.get(new_onset, '?')
            QMessageBox.information(self, "Undo", f"Restored {new_onset} to {new_offset_val}, from {old_onset} to {old_offset}")
            return
            
        # Cas validation simple (accept/reject) : restaurer le statut précédent
        if isinstance(action, tuple) and len(action) == 2:
            onset, prev_status = action
            # Restore previous validation status
            self.timeline_canvas.onset_validations[onset] = prev_status
            # Remove from event_status if it was there
            if hasattr(self, 'event_status') and onset in self.event_status:
                del self.event_status[onset]
            # Refresh view
            self.update_onset_filter()
            self.update_onset_info()
            self.redraw()
            QMessageBox.information(self, "Undo", f"Restored status to '{prev_status}' for onset {onset}")
            return
            
        # Cas par défaut
        QMessageBox.warning(self, "Undo", "Unknown action type, cannot undo.")

    def accept_and_toggle_offset(self):
    # Toggle offset validation widget
        if self.offset_validation_widget.isVisible():
            self.offset_validation_widget.hide()
            self.accept_btn.setEnabled(True)
            self.edit_btn.setEnabled(True)
            self.reject_btn.setEnabled(True)
            self.setFocus()
        # Cacher aussi le panneau d'édition si visible
            if hasattr(self, 'offset_edit_widget') and self.offset_edit_widget.isVisible():
                self.offset_edit_widget.hide()
            return
    # Sinon, ouvrir le menu et désactiver les autres boutons
        self.offset_validation_widget.show()
        self.accept_btn.setEnabled(True)
        self.edit_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)
        self.setFocus()

    def open_accept_menu(self, onset):
        self.current_onset_for_menu = onset
        self.menu_open = True
        self.offset_validation_widget.show()
        self.update_buttons_state(onset)

    def validate_accept(self):
        onset = self.current_onset_for_menu
        prev_status = self.timeline_canvas.onset_validations.get(onset, 'pending')
        if prev_status != 'accepted':
            self.undo_stack.append((onset, prev_status))
        self.timeline_canvas.onset_validations[onset] = 'accepted'
        self.menu_open = False
        self.offset_validation_widget.hide()
        self.update_buttons_state(onset)
        self.redraw()
        self.setFocus()  # Restore focus to main window

    def close_accept_menu(self):
        self.menu_open = False
        self.offset_validation_widget.hide()
        self.update_buttons_state(self.current_onset_for_menu)

    def redraw(self):
        self.timeline_canvas.plot_motion_energy_preserve_view(
            self.motion_energy,
            self.onsets,
            self.onset_types,
            self.timeline_canvas.event_offsets,
            getattr(self, 'event_status', None)
        )
        self.timeline_canvas.update_timeline()
        self.update_onset_info()
        self.update_performance_display()

    def update_buttons_state(self, onset):
        # Placeholder: implement logic if you want to enable/disable buttons based on onset
        pass

    # Dans init_ui, branche les boutons :
    # self.accept_btn.clicked.connect(lambda: self.open_accept_menu(self.onsets[self.current_onset_idx]))
    # self.validate_offset_btn.clicked.disconnect() puis self.validate_offset_btn.clicked.connect(self.validate_accept)
    # Pour fermer le menu offset, tu peux ajouter un bouton "Fermer" dans offset_validation_widget qui appelle self.close_accept_menu

    def export_project_mf(self):
        import pandas as pd
        # Crée le DataFrame de base
        n_frames = len(self.motion_energy) if self.motion_energy is not None else 0
        df = pd.DataFrame({
            'frame_idx': range(n_frames),
            'motion_energy': self.motion_energy if self.motion_energy is not None else [],
            'active': [0]*n_frames,
            'twitch': [0]*n_frames,
            'complex': [0]*n_frames,
            'status': ['']*n_frames,
            'score': [0]*n_frames
        })
        # Remplit les colonnes par événement validé
        for onset in self.onsets:
            offset = self.timeline_canvas.event_offsets.get(onset, onset)
            event_type = self.onset_types.get(onset, '')
            status = self.timeline_canvas.onset_validations.get(onset, 'pending')
            # Score selon la logique existante
            if status == 'accepted':
                score = 1
            elif status == 'rejected':
                score = -1
            elif status == 'edited':
                score = 0.5
            elif status == 'manually added':
                score = 0
            else:
                score = 0
            # Remplit la colonne binaire de onset à offset inclus
            if event_type in ('active', 'twitch', 'complex'):
                df.loc[onset:offset, event_type] = 1
            # Remplit status et score à l'onset
            df.at[onset, 'status'] = status
            df.at[onset, 'score'] = score
        # Sauvegarde le fichier
        fname, _ = QFileDialog.getSaveFileName(self, 'Export Project (MF)', '', 'Excel (*.xlsx);;CSV (*.csv)')
        if fname:
            if not fname.endswith('.xlsx'):
                fname += '.xlsx'
            df.to_excel(fname, index=False)

    def choose_export_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Export Directory')
        if dir_path:
            self.export_path_lineedit.setText(dir_path)

    def export_all_outputs(self):
        import pandas as pd
        import os, json
        dir_path = self.export_path_lineedit.text()
        if not dir_path:
            QMessageBox.warning(self, "Export", "Please choose an export directory first.")
            return
        overlaps = self.check_for_overlaps()
        if overlaps:
            overlap_str = "\n".join([f"{a1}-{b1} overlaps {a2}-{b2}" for a1, b1, a2, b2 in overlaps])
            reply = QMessageBox.question(
                self,
                "Overlapping Events",
                f"The following events overlap:\n\n{overlap_str}\n\nDo you want to export anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        output_dir = os.path.join(dir_path, 'mousecraft_output')
        os.makedirs(output_dir, exist_ok=True)
        # If no classification table is present, build a minimal input table from motion length
        if not hasattr(self, 'input_classification_df') or self.input_classification_df is None:
            import pandas as pd
            n_frames = len(self.motion_energy) if self.motion_energy is not None else (max(self.onsets)+1 if self.onsets else 0)
            input_df = pd.DataFrame({'frame_idx': list(range(n_frames))})
            has_pending = False
            manual_only_export = True
        else:
            input_df = self.input_classification_df.copy()
            has_pending = any(self.timeline_canvas.onset_validations.get(onset, 'pending') == 'pending' for onset in self.onsets)
            manual_only_export = False
        suffix = '_pending' if has_pending else '_final'
        mf_df = input_df.copy()
        n_frames = len(mf_df)
        for col in ['active', 'twitch', 'complex']:
            # if col not in mf_df.columns:
            mf_df[col] = 0
        mf_df['status'] = [''] * n_frames
        mf_df['score'] = [''] * n_frames
        # Pour chaque événement, pose les 1 de onset à offset selon la logique demandée
        for onset in self.onsets:
            event_type = self.onset_types.get(onset, '')
            status = self.timeline_canvas.onset_validations.get(onset, 'pending')
            # Always use the current (possibly edited) onset and offset
            start = min(onset, self.timeline_canvas.event_offsets.get(onset, onset))
            end = max(onset, self.timeline_canvas.event_offsets.get(onset, onset))
            # For export: make offset inclusive; ensure at least one-frame span
            export_offset = end + 1 if end > start else (start + 1)
            if event_type in ['active', 'twitch', 'complex']:
                mf_df.loc[start:end, event_type] = 1
            # Write status and score only at the onset frame
            if manual_only_export:
                score = 0
                status_label = 'manually added'
            elif status == 'accepted':
                score = 1
                status_label = 'accepted'
            elif status == 'rejected':
                score = -1
                status_label = 'rejected'
            elif status == 'edited':
                if self.event_status.get(onset, 0.5) == 1:
                    score = 1
                else:
                    score = 0.5
                status_label = 'edited'
            elif status == 'manually added':
                score = 0
                status_label = 'manually added'
            else:
                score = 0
                status_label = status
            mf_df.at[onset, 'status'] = status_label
            mf_df.at[onset, 'score'] = score
        mf_path = os.path.join(output_dir, f"validation_MF{suffix}.xlsx")
        created_files = []
        try:
            mf_df.to_excel(mf_path, index=False)
            created_files.append(mf_path)
        except PermissionError:
            # Try saving in a new folder
            alt_dir = os.path.join(os.path.dirname(output_dir), "mousecraft_output_2")
            os.makedirs(alt_dir, exist_ok=True)
            alt_path = os.path.join(alt_dir, os.path.basename(mf_path))
            mf_df.to_excel(alt_path, index=False)
            created_files.append(alt_path)
            QMessageBox.warning(
                self, "Export Warning",
                "Careful: I could not overwrite the file because it is open in Excel (or another program).\n"
                f"The output was saved in {alt_dir} instead."
            )
        except Exception:
            # Fallback to CSV in the same directory
            mf_path_csv = os.path.join(output_dir, f"validation_MF{suffix}.csv")
            try:
                mf_df.to_csv(mf_path_csv, index=False)
                created_files.append(mf_path_csv)
            except Exception as e2:
                QMessageBox.critical(self, "Export Error", f"Failed to save validation_MF: {e2}")
        # Sauvegarde aussi en numpy uniquement
        import numpy as np
        if len(mf_df) > 0:
            mf_npy_path = os.path.splitext(created_files[-1])[0] + ".npy" if created_files else os.path.join(output_dir, f"validation_MF{suffix}.npy")
            try:
                np.save(mf_npy_path, mf_df.to_numpy())
                created_files.append(mf_npy_path)
            except Exception:
                pass
        export_events = []
        for onset in self.onsets:
            offset = self.timeline_canvas.event_offsets.get(onset, onset)
            event_type = self.onset_types.get(onset, '')
            status = self.timeline_canvas.onset_validations.get(onset, 'pending')
            if status == 'accepted':
                score = 1
            elif status == 'rejected':
                score = -1
            elif status == 'edited':
                score = 0.5
            elif status == 'manually added':
                score = 0
            else:
                score = 0
            export_events.append([onset, offset, event_type, status, score])
        hf_df = pd.DataFrame(export_events, columns=pd.Index(['onset', 'offset', 'event_type', 'status', 'score']))
        hf_path = os.path.join(output_dir, f"validation_HF{suffix}.xlsx")
        try:
            hf_df.to_excel(hf_path, index=False)
            created_files.append(hf_path)
        except Exception:
            # Fallback to CSV
            hf_path_csv = os.path.join(output_dir, f"validation_HF{suffix}.csv")
            try:
                hf_df.to_csv(hf_path_csv, index=False)
                created_files.append(hf_path_csv)
            except Exception as e2:
                QMessageBox.critical(self, "Export Error", f"Failed to save validation_HF: {e2}")
        # Sauvegarde aussi en numpy uniquement
        if len(hf_df) > 0:
            hf_npy_path = os.path.splitext(created_files[-1])[0] + ".npy" if created_files else os.path.join(output_dir, f"validation_HF{suffix}.npy")
            try:
                np.save(hf_npy_path, hf_df.to_numpy())
                created_files.append(hf_npy_path)
            except Exception:
                pass
        plot_path = os.path.join(output_dir, f"validation_comparison_plot{suffix}.png")
        try:
            self.create_validation_comparison_plot(export_events, save_path=plot_path)
            created_files.append(plot_path)
        except Exception:
            pass
        # Add final classification plot only if no pending events
        if not has_pending and not manual_only_export:
            # Remove all _pending files in output_dir
            for fname in os.listdir(output_dir):
                if '_pending' in fname:
                    try:
                        os.remove(os.path.join(output_dir, fname))
                    except Exception as e:
                        print(f"Could not remove {fname}: {e}")
            final_plot_path = os.path.join(output_dir, f"final_classification_plot{suffix}.png")
            try:
                self.create_final_classification_plot(export_events, save_path=final_plot_path)
                created_files.append(final_plot_path)
            except Exception:
                pass
        # Export pie chart of event status distribution
        pie_chart_path = os.path.join(output_dir, f"validation_status_pie{suffix}.png")
        try:
            self.create_validation_pie_chart(export_events, save_path=pie_chart_path)
            created_files.append(pie_chart_path)
        except Exception:
            pass
        # Export metrics JSON
        metrics_path = os.path.join(output_dir, f"validation_metrics{suffix}.json")
        if manual_only_export:
            try:
                with open(metrics_path, 'w') as f:
                    json.dump({'mode': 'manual_only', 'note': 'All events are manually added; no performance metrics.'}, f, indent=2)
                created_files.append(metrics_path)
            except Exception:
                pass
        else:
            true_positives = sum(1 for event in export_events if event[3] in ('accepted', 'edited'))
            false_positives = sum(1 for event in export_events if event[3] == 'rejected')
            total = true_positives + false_positives
            precision = true_positives / total if total > 0 else 0
            results = {
                'performance_metrics': {
                    'true_positives': true_positives,
                    'false_positives': false_positives
                },
                'precision': precision,
                'total_annotated': total
            }
            try:
                with open(metrics_path, 'w') as f:
                    json.dump(results, f, indent=2)
                created_files.append(metrics_path)
            except Exception:
                pass
        # If a second signal is present, export its annotations separately with _second_signal suffix
        if hasattr(self, 'timeline_canvas2') and self.timeline_canvas2.isVisible() and hasattr(self, 'onsets2') and self.onsets2 is not None:
            # Build a minimal frame table for second signal
            n_frames2 = len(getattr(self.timeline_canvas2, 'motion_energy', [])) if getattr(self.timeline_canvas2, 'motion_energy', None) is not None else (max(self.onsets2)+1 if self.onsets2 else 0)
            try:
                import pandas as pd
                mf2_df = pd.DataFrame({'frame_idx': list(range(n_frames2))})
                for col in ['active', 'twitch', 'complex']:
                    mf2_df[col] = 0
                mf2_df['status'] = [''] * n_frames2
                mf2_df['score'] = [''] * n_frames2
                # Fill binary columns based on onsets2/event_types2 and offsets (inclusive export)
                for onset in self.onsets2:
                    event_type2 = self.onset_types2.get(onset, '')
                    off2 = self.timeline_canvas2.event_offsets.get(onset, onset)
                    start2 = min(onset, off2)
                    end2 = max(onset, off2)
                    export_off2 = end2 + 1 if end2 > start2 else (start2 + 1)
                    if event_type2 in ['active', 'twitch', 'complex']:
                        mf2_df.loc[start2:end2, event_type2] = 1
                    mf2_df.at[onset, 'status'] = 'manually added'
                    mf2_df.at[onset, 'score'] = 0
                # Write MF for second signal
                mf2_path = os.path.join(output_dir, f"validation_MF_input_1{suffix}.xlsx")
                try:
                    mf2_df.to_excel(mf2_path, index=False)
                    created_files.append(mf2_path)
                except Exception:
                    mf2_path_csv = os.path.join(output_dir, f"validation_MF_input_1{suffix}.csv")
                    try:
                        mf2_df.to_csv(mf2_path_csv, index=False)
                        created_files.append(mf2_path_csv)
                    except Exception:
                        pass
                # HF export for second signal
                export_events2 = []
                for onset in self.onsets2:
                    off2 = self.timeline_canvas2.event_offsets.get(onset, onset)
                    event_type2 = self.onset_types2.get(onset, '')
                    # Always manual for second when motion-only
                    status2 = 'manually added'
                    score2 = 0
                    s2 = min(onset, off2)
                    e2 = max(onset, off2)
                    export_off2 = e2 + 1 if e2 > s2 else (s2 + 1)
                    export_events2.append([onset, export_off2, event_type2, status2, score2])
                hf2_df = pd.DataFrame(export_events2, columns=pd.Index(['onset', 'offset', 'event_type', 'status', 'score']))
                hf2_path = os.path.join(output_dir, f"validation_HF_input_1{suffix}.xlsx")
                try:
                    hf2_df.to_excel(hf2_path, index=False)
                    created_files.append(hf2_path)
                except Exception:
                    hf2_path_csv = os.path.join(output_dir, f"validation_HF_input_1{suffix}.csv")
                    try:
                        hf2_df.to_csv(hf2_path_csv, index=False)
                        created_files.append(hf2_path_csv)
                    except Exception:
                        pass
                # NPYs for second signal
                try:
                    import numpy as np
                    if len(mf2_df) > 0:
                        np.save(os.path.splitext(mf2_path)[0] + '.npy' if 'mf2_path' in locals() else os.path.join(output_dir, f"validation_MF_second_signal{suffix}.npy"), mf2_df.to_numpy())
                        created_files.append((os.path.splitext(mf2_path)[0] + '.npy') if 'mf2_path' in locals() else os.path.join(output_dir, f"validation_MF_second_signal{suffix}.npy"))
                    if len(hf2_df) > 0:
                        np.save(os.path.splitext(hf2_path)[0] + '.npy' if 'hf2_path' in locals() else os.path.join(output_dir, f"validation_HF_second_signal{suffix}.npy"), hf2_df.to_numpy())
                        created_files.append((os.path.splitext(hf2_path)[0] + '.npy') if 'hf2_path' in locals() else os.path.join(output_dir, f"validation_HF_second_signal{suffix}.npy"))
                except Exception:
                    pass
            except Exception:
                pass
        # Build message from actually created files
        msg = f"Exported to: {output_dir}\n\n" + "\n".join(f"- {os.path.basename(p)}" for p in created_files)
        QMessageBox.information(self, "Export", msg)
        # Après export, reset le flag
        self.unsaved_changes = False

    def create_final_classification_plot(self, export_events, save_path=None):
        """Create a plot showing only the final classification (excluding rejected events) and motion energy. No color markers."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        if self.motion_energy is None:
            QMessageBox.warning(self, "Warning", "No motion energy data available for final classification plot")
            return
        # Filter out rejected events
        final_events = [e for e in export_events if e[3] != 'rejected']
        fig, ax = plt.subplots(figsize=(15, 3.5))
        ax.plot(np.arange(len(self.motion_energy)), self.motion_energy, color='blue', linewidth=1, alpha=0.7)
        for event in final_events:
            onset, offset, event_type, status, score = event
            if event_type == 'twitch':
                ax.axvspan(onset, offset, color='purple', alpha=0.6)
            elif event_type == 'active':
                ax.axvspan(onset, offset, color='yellow', alpha=0.6)
        ax.set_title('Final Motion Energy Classification (Rejected Events Hidden)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Motion Energy', fontsize=12)
        ax.set_xlim(0, len(self.motion_energy))
        ax.set_ylim(0, max(self.motion_energy) * 1.2)
        ax.grid(True, alpha=0.3)
        legend_elements = [
            Line2D([0], [0], color='purple', alpha=0.6, linewidth=10, label='Twitch'),
            Line2D([0], [0], color='yellow', alpha=0.6, linewidth=10, label='Active'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def closeEvent(self, event):
        if getattr(self, 'unsaved_changes', False):
            reply = QMessageBox.question(self, "Save before exit?", "Do you want to save your work before exiting?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                # Ouvre le sélecteur d'export si pas déjà choisi
                if not self.export_path_lineedit.text():
                    self.choose_export_path()
                    # Si l'utilisateur annule le choix du dossier, ne pas exporter
                    if not self.export_path_lineedit.text():
                        event.ignore()
                        return
                self.export_all_outputs()
                event.accept()
                return
            else:
                event.accept()
                return
        event.accept()

    def check_for_overlaps(self):
        """Return a list of tuples (onset1, offset1, onset2, offset2) for all overlapping event pairs, ignoring rejected events."""
        overlaps = []
        # Only consider events that are not rejected
        valid_onsets = [onset for onset in self.onsets if self.timeline_canvas.onset_validations.get(onset, 'pending') not in ('rejected',)]
        events = [(onset, self.timeline_canvas.event_offsets.get(onset, onset)) for onset in valid_onsets]
        events = sorted(events, key=lambda x: x[0])
        for i in range(len(events)):
            onset1, offset1 = events[i]
            for j in range(i+1, len(events)):
                onset2, offset2 = events[j]
                # If onset2 < offset1, they overlap
                if onset2 < offset1:
                    overlaps.append((onset1, offset1, onset2, offset2))
                else:
                    break
        return overlaps

    def show_firework_animation(self):
        """Show a firework GIF animation for 5 seconds with a congratulatory message below, with a solid dialog background."""
        print("🎆 Firework animation triggered! 🎆")
        from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtGui import QMovie
        dialog = QDialog(self)
        dialog.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        # Do NOT set Qt.WA_TranslucentBackground
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Firework GIF
        firework_label = QLabel(dialog)
        firework_label.setAlignment(Qt.AlignCenter)
        firework_path = os.path.join(SCRIPT_DIR, "resources", "Fireworks.gif")
        print(f"Firework path: {firework_path}")
        print(f"Firework file exists: {os.path.exists(firework_path)}")
        movie = QMovie(firework_path)
        firework_label.setMovie(movie)
        firework_label.setFixedSize(400, 400)
        movie.start()
        layout.addWidget(firework_label, alignment=Qt.AlignCenter)
        # Message below
        message_label = QLabel("Congratulations! All events have been validated.", dialog)
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #333; background: white;")
        layout.addWidget(message_label, alignment=Qt.AlignCenter)
        dialog.setLayout(layout)
        dialog.resize(420, 470)
        dialog.show()
        QTimer.singleShot(5000, dialog.accept)

    def set_event_status(self, onset, status):
        prev_status = self.timeline_canvas.onset_validations.get(onset, 'pending')
        self.timeline_canvas.onset_validations[onset] = status
        # Start autosave if status changed from pending to something else
        if prev_status == 'pending' and status != 'pending':
            self.maybe_start_auto_save()
        # Stop autosave if there are no more pending events
        self.maybe_stop_auto_save()
        # ... any other logic ...
        # Check if there are no more pending events
        if not any(self.timeline_canvas.onset_validations.get(o, 'pending') == 'pending' for o in self.onsets):
            self.show_firework_animation()
        # ... rest of the function ...

    def show_mouse_animation(self):
        """Mouse moves from far left to center, stops, says something with a bubble for 3s, then continues to far right."""
        print("🎉 Mouse animation triggered! 🎉")
        from PyQt5.QtWidgets import QDialog, QLabel
        from PyQt5.QtCore import Qt, QTimer, QSize
        from PyQt5.QtGui import QMovie, QPixmap
        import os
        import random
        dialog_width = 1000
        dialog_height = 300
        mouse_path = os.path.join(SCRIPT_DIR, "resources", "mouse.png")
        bubble_path = os.path.join(SCRIPT_DIR, "resources", "bulle de parole.wepb")  # Use the correct bubble file
        print(f"Mouse path: {mouse_path}")
        print(f"Mouse file exists: {os.path.exists(mouse_path)}")
        print(f"Bubble path: {bubble_path}")
        print(f"Bubble file exists: {os.path.exists(bubble_path)}")
        # Load mouse image to get its size
        if os.path.exists(mouse_path):
            from PIL import Image
            im = Image.open(mouse_path)
            mouse_width, mouse_height = im.size
        else:
            mouse_width, mouse_height = 239, 234  # fallback
        max_mouse_height = int(dialog_height * 0.8)
        scale_factor = min(1.0, max_mouse_height / mouse_height)
        scaled_mouse_width = int(mouse_width * scale_factor)
        scaled_mouse_height = int(mouse_height * scale_factor)
        # Load bubble image to get its size
        if os.path.exists(bubble_path):
            from PIL import Image
            im_bubble = Image.open(bubble_path)
            bubble_width, bubble_height = im_bubble.size
        else:
            bubble_width, bubble_height = 180, 100  # fallback
        max_bubble_width = int(scaled_mouse_width * 1.5)
        scale_bubble = min(1.0, max_bubble_width / bubble_width)
        scaled_bubble_width = int(bubble_width * scale_bubble)
        scaled_bubble_height = int(bubble_height * scale_bubble)
        messages = [
            "Congrats, you're doing a good job!",
            "A bit more, you can do it!",
            "Ahah, more and more twitch!"
        ]
        chosen_message = random.choice(messages)
        dialog = QDialog(self)
        dialog.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        dialog.setModal(True)
        dialog.resize(dialog_width, dialog_height)
        # Mouse label
        mouse_label = QLabel(dialog)
        mouse_label.resize(scaled_mouse_width, scaled_mouse_height)
        mouse_label.setAttribute(Qt.WA_TranslucentBackground)
        if os.path.splitext(mouse_path)[1].lower() in [".gif"]:
            movie = QMovie(mouse_path)
            movie.setScaledSize(QSize(scaled_mouse_width, scaled_mouse_height))
            mouse_label.setMovie(movie)
            movie.start()
        else:
            pixmap = QPixmap(mouse_path)
            mouse_label.setPixmap(pixmap.scaled(scaled_mouse_width, scaled_mouse_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # Bubble label
        bubble_label = QLabel(dialog)
        bubble_label.resize(scaled_bubble_width, scaled_bubble_height)
        bubble_label.setAttribute(Qt.WA_TranslucentBackground)
        if os.path.exists(bubble_path):
            bubble_pixmap = QPixmap(bubble_path)
            bubble_label.setPixmap(bubble_pixmap.scaled(scaled_bubble_width, scaled_bubble_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # Message label (child of bubble_label)
        message_label = QLabel(bubble_label)
        message_label.setText(f"<div style='text-align:center; font-size:14px; font-weight:bold; color:#222;'>{chosen_message}</div>")
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setFixedWidth(scaled_bubble_width - 20)
        message_label.setFixedHeight(scaled_bubble_height - 20)
        message_label.move(10, 10)
        message_label.setStyleSheet("background: transparent;")
        # Start/end positions
        start_x = -scaled_mouse_width
        center_x = (dialog_width - scaled_mouse_width) // 2
        end_x = dialog_width
        y_mouse = (dialog_height - scaled_mouse_height) // 2
        y_bubble = y_mouse - scaled_bubble_height - 10
        # Animation timing
        move1_steps = 60
        move2_steps = 60
        interval = 30  # ms per step
        pause_duration = 3000  # 3 seconds
        # Hide bubble initially
        bubble_label.hide()
        def move_to_center(step=0):
            if step > move1_steps:
                # Show bubble and message, pause
                bubble_label.show()
                bubble_x = center_x + scaled_mouse_width // 2
                bubble_label.move(bubble_x, y_bubble)
                QTimer.singleShot(pause_duration, lambda: after_pause())
                return
            x = int(start_x + (center_x - start_x) * step / move1_steps)
            mouse_label.move(x, y_mouse)
            QTimer.singleShot(interval, lambda: move_to_center(step + 1))
        def after_pause():
            bubble_label.hide()
            move_to_right()
        def move_to_right(step=0):
            if step > move2_steps:
                dialog.accept()
                return
            x = int(center_x + (end_x - center_x) * step / move2_steps)
            mouse_label.move(x, y_mouse)
            QTimer.singleShot(interval, lambda: move_to_right(step + 1))
        move_to_center()

    def frame_spinbox_changed(self, value):
        # Jump to the frame entered by the user
        self.current_frame = value
        self.frame_slider.setValue(value)
        self.show_frame(value)
        self.timeline_canvas.current_frame = value
        self.timeline_canvas.update_timeline()
        self.update_onset_info()

    def create_validation_pie_chart(self, export_events, save_path=None):
        """Create and save a pie chart showing the percentage of accepted, edited (<X, >X), pending, rejected, and manually added events."""
        import matplotlib.pyplot as plt
        from collections import Counter
        edit_threshold = getattr(self, 'edit_threshold', 5)
        status_buckets = []
        for event in export_events:
            status = event[3]
            if status == 'edited':
                status_buckets.append('edited')
            else:
                status_buckets.append(status)
        counter = Counter(status_buckets)
        labels = []
        sizes = []
        colors = []
        color_map = {
            'accepted': '#4CAF50',      # Green
            'edited': '#FFA726',        # Orange
            'pending': '#90A4AE',       # Gray-blue
            'rejected': '#F44336',      # Red
            'manually added': '#00CED1' # Cyan
        }
        for status in ['accepted', 'edited', 'pending', 'rejected', 'manually added']:
            if counter[status] > 0:
                labels.append(f"{status.capitalize()} ({counter[status]})")
                sizes.append(counter[status])
                colors.append(color_map.get(status, '#BDBDBD'))
        if not sizes:
            return  # Nothing to plot
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
        plt.title('Event Validation Status Distribution', fontsize=15, fontweight='bold')
        plt.axis('equal')
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    def setup_auto_save(self, interval_minutes=20):
        """Set up the autosave timer but do not start it yet."""
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self.auto_save)
        self._auto_save_started = False  # Track if timer has started
        self._auto_save_interval = interval_minutes * 60 * 1000

    def maybe_start_auto_save(self):
        """Start autosave timer if at least one event is not pending and timer not already started."""
        if not hasattr(self, 'auto_save_timer'):
            self.setup_auto_save()
        if not getattr(self, '_auto_save_started', False):
            if hasattr(self, 'onsets') and self.onsets:
                non_pending = any(
                    self.timeline_canvas.onset_validations.get(o, 'pending') != 'pending'
                    for o in self.onsets
                )
                if non_pending:
                    self.auto_save_timer.start(self._auto_save_interval)
                    self._auto_save_started = True

    def auto_save(self):
        # Only autosave if at least one event is not pending
        if hasattr(self, 'onsets') and self.onsets:
            non_pending = any(
                self.timeline_canvas.onset_validations.get(o, 'pending') != 'pending'
                for o in self.onsets
            )
            if not non_pending:
                return  # Do not autosave if all are pending
        dir_path = self.export_path_lineedit.text()
        if not dir_path:
            QMessageBox.information(
                self, "Auto-Save",
                "Please choose an export directory for auto-save to work."
            )
            self.choose_export_path()
            dir_path = self.export_path_lineedit.text()
            if not dir_path:
                return  # User cancelled
        self.export_all_outputs()
        print("Auto-saved project at", dir_path)

    def check_mouse_milestones(self):
        total = len(self.onsets)
        non_pending = sum(
            1 for o in self.onsets
            if self.timeline_canvas.onset_validations.get(o, 'pending') not in ['pending']
        )
        print(f"Mouse milestone check: total={total}, non_pending={non_pending}, half_shown={getattr(self, '_mouse_half_shown', False)}")
        # 1. Always show at halfway
        halfway = total // 2
        if not getattr(self, '_mouse_half_shown', False) and non_pending >= halfway and total > 0:
            self.show_mouse_animation()
            self._mouse_half_shown = True
            # After halfway, set up additional milestones
            # 1 for 100, 2 for 200, 3 for 300+
            if total >= 300:
                n_milestones = 3
            elif total >= 200:
                n_milestones = 2
            elif total >= 100:
                n_milestones = 1
            else:
                n_milestones = 0
            # Evenly space milestones after halfway, before total
            if n_milestones > 0:
                self._mouse_milestones = set([
                    halfway + ((i+1)*(total-halfway)//(n_milestones+1)) for i in range(n_milestones)
                ])
            else:
                self._mouse_milestones = set()
            self._mouse_milestones_shown = set()
            return
        # 2. After halfway, show at milestone points
        if getattr(self, '_mouse_half_shown', False) and hasattr(self, '_mouse_milestones'):
            for milestone in self._mouse_milestones:
                if non_pending >= milestone and milestone not in self._mouse_milestones_shown:
                    self.show_mouse_animation()
                    self._mouse_milestones_shown.add(milestone)

    def check_all_validated_and_show_firework(self):
        if not hasattr(self, '_firework_shown') or not self._firework_shown:
            if all(self.timeline_canvas.onset_validations.get(o, 'pending') != 'pending' for o in self.onsets):
                self.show_firework_animation()
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "Congratulations!", "Congratulations! All events have been validated.")
                self._firework_shown = True

    def maybe_stop_auto_save(self):
        """Stop autosave timer if there are no more pending events."""
        if hasattr(self, 'auto_save_timer') and getattr(self, '_auto_save_started', False):
            if hasattr(self, 'onsets') and self.onsets:
                all_non_pending = all(
                    self.timeline_canvas.onset_validations.get(o, 'pending') != 'pending'
                    for o in self.onsets
                )
                if all_non_pending:
                    self.auto_save_timer.stop()
                    self._auto_save_started = False
                    
                    
    def start_mouse_timer(self):
        """Lance un timer pour afficher la souris toutes les 20 minutes."""
        self.mouse_timer = QTimer(self)
        self.mouse_timer.timeout.connect(self.show_mouse_animation)  # Appelle l'animation
        self.mouse_timer.start(20 * 60 * 1000)  # 20 minutes en millisecondes


    def check_total_overlap_and_prompt(self, new_onset, new_offset, exclude_onsets=None):
        """Check for totally overlapped events and prompt to reject them. Exclude the event(s) being edited/added if needed."""
        if exclude_onsets is None:
            exclude_onsets = []
        elif not isinstance(exclude_onsets, (list, tuple, set)):
            exclude_onsets = [exclude_onsets]
        overlapped = []
        for other_onset in self.onsets:
            if other_onset in exclude_onsets:
                continue
            other_offset = self.timeline_canvas.event_offsets.get(other_onset, other_onset)
            # Check if other event is totally inside new event
            if new_onset < other_onset and new_offset > other_offset:
                overlapped.append((other_onset, other_offset, self.onset_types.get(other_onset, 'unknown')))
        for onset, offset, event_type in overlapped:
            reply = QMessageBox.question(
                self,
                "Total Overlap Detected",
                f"The event {event_type} ({onset}-{offset}) is totally overlapped by your new/edited event.\nDo you want to reject it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.timeline_canvas.onset_validations[onset] = 'rejected'
                self.update_onset_filter()
                self.update_onset_info()
                self.redraw()
                # Go to next non-rejected onset
                idx = self.onsets.index(onset)
                next_idx = idx + 1
                while next_idx < len(self.onsets):
                    next_onset = self.onsets[next_idx]
                    if self.timeline_canvas.onset_validations.get(next_onset, 'pending') != 'rejected':
                        self.goto_onset(next_idx)
                        break
                    next_idx += 1

    def check_total_overlap_and_prompt_for(self, onsets_list, offsets_map, onset_types, validations, new_onset, new_offset, exclude_onsets=None):
        """Variant of total-overlap check for arbitrary stores (used for second signal)."""
        if exclude_onsets is None:
            exclude_onsets = []
        elif not isinstance(exclude_onsets, (list, tuple, set)):
            exclude_onsets = [exclude_onsets]
        overlapped = []
        for other_onset in onsets_list:
            if other_onset in exclude_onsets:
                continue
            other_offset = offsets_map.get(other_onset, other_onset)
            if new_onset < other_onset and new_offset > other_offset:
                overlapped.append((other_onset, other_offset, onset_types.get(other_onset, 'unknown')))
        for onset, offset, event_type in overlapped:
            reply = QMessageBox.question(
                self,
                "Total Overlap Detected",
                f"The event {event_type} ({onset}-{offset}) is totally overlapped by your new/edited event.\nDo you want to reject it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                validations[onset] = 'rejected'
                # No redraw of primary timeline here

    def show_change_type_dropdown(self):
        # Remove any existing dropdown
        if hasattr(self, 'change_type_dropdown') and self.change_type_dropdown is not None:
            self.change_type_dropdown.hide()
            self.change_type_dropdown.deleteLater()
        # Create dropdown
        from PyQt5.QtWidgets import QComboBox
        self.change_type_dropdown = QComboBox(self)
        self.change_type_dropdown.addItems(self.available_event_types)
        # Set current type as selected
        if hasattr(self, 'filtered_onsets') and self.filtered_onsets:
            current_onset = self.filtered_onsets[self.current_onset_idx]
        else:
            current_onset = self.onsets[self.current_onset_idx]
        current_type = self.onset_types.get(current_onset, "twitch")
        idx = self.change_type_dropdown.findText(current_type)
        if idx >= 0:
            self.change_type_dropdown.setCurrentIndex(idx)
        # Place dropdown next to the button
        btn_pos = self.change_type_btn.mapToGlobal(self.change_type_btn.rect().bottomLeft())
        self.change_type_dropdown.move(self.mapFromGlobal(btn_pos))
        self.change_type_dropdown.showPopup()
        self.change_type_dropdown.activated[str].connect(lambda new_type: self.change_event_type(new_type, current_onset))
        self.change_type_dropdown.show()
    def change_event_type(self, new_type, onset):
        self.onset_types[onset] = new_type
        self.timeline_canvas.onset_types[onset] = new_type
        self.redraw()
        self.update_onset_info()
        if hasattr(self, 'change_type_dropdown') and self.change_type_dropdown is not None:
            self.change_type_dropdown.hide()
            self.change_type_dropdown.deleteLater()
            self.change_type_dropdown = None

    def open_add_type_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add a new event type")
        layout = QVBoxLayout(dialog)
        form_layout = QGridLayout()
        name_label = QLabel("Type name:")
        name_input = QLineEdit()
        form_layout.addWidget(name_label, 0, 0)
        form_layout.addWidget(name_input, 0, 1)
        color_label = QLabel("Color:")
        color_preview = QLabel("     ")
        color_preview.setStyleSheet("background: #888; border: 1px solid #444;")
        pick_btn = QPushButton("Pick color")
        form_layout.addWidget(color_label, 1, 0)
        form_layout.addWidget(color_preview, 1, 1)
        form_layout.addWidget(pick_btn, 1, 2)
        layout.addLayout(form_layout)
        btns = QHBoxLayout()
        ok_btn = QPushButton("Add")
        cancel_btn = QPushButton("Cancel")
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        selected_color = {'value': '#888888'}

        def pick_color():
            color = QColorDialog.getColor(QColor(selected_color['value']), self, "Choose color")
            if color.isValid():
                selected_color['value'] = color.name()
                color_preview.setStyleSheet(f"background: {selected_color['value']}; border: 1px solid #444;")

        pick_btn.clicked.connect(pick_color)

        def accept():
            name = name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "Invalid name", "Please enter a type name.")
                return
            key = name.lower()
            if key in self.available_event_types:
                QMessageBox.information(self, "Already exists", f"Type '{name}' already exists.")
                return
            # Save
            self.available_event_types.append(key)
            self.event_type_colors[key] = selected_color['value']
            # Update UI combos
            # Manual add combo (preserve placeholder at index 0)
            current_placeholder = self.event_type_combo.itemText(0)
            self.event_type_combo.clear()
            self.event_type_combo.addItem(current_placeholder)
            self.event_type_combo.addItems(self.available_event_types)
            self.event_type_combo.setCurrentIndex(0)
            # Onset filter combo
            current_status_idx = self.onset_filter_combo.currentIndex()
            self.onset_filter_combo.clear()
            self.onset_filter_combo.addItems(["All"] + [t.capitalize() for t in self.available_event_types])
            # Keep selection if possible
            if current_status_idx < self.onset_filter_combo.count():
                self.onset_filter_combo.setCurrentIndex(current_status_idx)
            # Change-type dropdown if currently open
            if hasattr(self, 'change_type_dropdown') and self.change_type_dropdown is not None:
                self.change_type_dropdown.clear()
                self.change_type_dropdown.addItems(self.available_event_types)
            # Also propagate mapping and visibility to both timelines
            if hasattr(self, 'timeline_canvas') and self.timeline_canvas is not None:
                self.timeline_canvas.event_type_colors = dict(self.event_type_colors)
                try:
                    if hasattr(self.timeline_canvas, 'visible_event_types') and self.timeline_canvas.visible_event_types is not None:
                        if key not in self.timeline_canvas.visible_event_types:
                            self.timeline_canvas.visible_event_types.append(key)
                except Exception:
                    pass
            if hasattr(self, 'timeline_canvas2') and self.timeline_canvas2 is not None:
                self.timeline_canvas2.event_type_colors = dict(self.event_type_colors)
                try:
                    if hasattr(self.timeline_canvas2, 'visible_event_types') and self.timeline_canvas2.visible_event_types is not None:
                        if key not in self.timeline_canvas2.visible_event_types:
                            self.timeline_canvas2.visible_event_types.append(key)
                except Exception:
                    pass
            dialog.accept()
            self.redraw()

        ok_btn.clicked.connect(accept)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec_()


def main(): # launches play button and then main mousecraft 
    import sys, subprocess, os
    from PyQt5.QtWidgets import QApplication

    # Path to startup script
    startup_script = os.path.join(os.path.dirname(__file__), "mousecraft.py")

    # Wait for startup to close
    subprocess.run([sys.executable, startup_script])

    # Now launch PyQt GUI
    app = QApplication(sys.argv)
    annotator = MotionAnnotator()
    annotator.pause_btn.raise_()
    annotator.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()