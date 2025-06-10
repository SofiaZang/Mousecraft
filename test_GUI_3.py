import sys
import cv2
import json
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLabel, QSlider, QFileDialog, QHBoxLayout, QSpinBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MotionAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Onset Editor")
        self.video_path = None
        self.motion_energy = None
        self.current_frame = 0
        self.classified_events = {}  # original classifications
        self.curated_events = {}     # editable copy
        self.onsets = []             # list of all onsets
        self.current_onset_idx = 0
        self.edit_mode = False       # track if editing is enabled

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def init_ui(self):
        self.image_label = QLabel("Video will appear here")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)

        # Motion control
        self.play_btn = QPushButton("Play ▶️")
        self.pause_btn = QPushButton("Pause ⏸")
        self.load_video_btn = QPushButton("Load Video")
        self.load_class_btn = QPushButton("Load Classifications")
        self.load_me_btn = QPushButton("Load Motion Energy")
        self.save_btn = QPushButton("Save Curated Onsets")

        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.load_video_btn.clicked.connect(self.load_video)
        self.load_class_btn.clicked.connect(self.load_classifications)
        self.load_me_btn.clicked.connect(self.load_motion_energy)
        self.save_btn.clicked.connect(self.save_curated_onsets)

        # Onset navigation
        self.prev_btn = QPushButton("← Prev Onset")
        self.next_btn = QPushButton("Next Onset →")
        self.shift_left_btn = QPushButton("Shift ←")
        self.shift_right_btn = QPushButton("Shift →")
        self.edit_btn = QPushButton("Edit Mode")

        self.shift_step_box = QSpinBox()
        self.shift_step_box.setMinimum(1)
        self.shift_step_box.setMaximum(30)
        self.shift_step_box.setValue(1)

        self.prev_btn.clicked.connect(self.prev_onset)
        self.next_btn.clicked.connect(self.next_onset)
        self.shift_left_btn.clicked.connect(lambda: self.shift_onset(-self.shift_step_box.value()))
        self.shift_right_btn.clicked.connect(lambda: self.shift_onset(self.shift_step_box.value()))
        self.edit_btn.clicked.connect(self.toggle_edit_mode)

        # Layouts
        nav_layout = QHBoxLayout()
        for btn in [self.play_btn, self.pause_btn, self.load_video_btn, self.load_class_btn, self.load_me_btn, self.save_btn]:
            nav_layout.addWidget(btn)

        onset_nav_layout = QHBoxLayout()
        for btn in [self.prev_btn, self.next_btn, self.shift_left_btn, self.shift_right_btn, self.edit_btn]:
            onset_nav_layout.addWidget(btn)
        onset_nav_layout.addWidget(QLabel("Step:"))
        onset_nav_layout.addWidget(self.shift_step_box)

        self.fig = Figure(figsize=(5, 1.5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider)
        layout.addLayout(onset_nav_layout)
        layout.addLayout(nav_layout)
        self.setLayout(layout)

        self.update_edit_mode_ui()

    def toggle_edit_mode(self):
        self.edit_mode = not self.edit_mode
        self.update_edit_mode_ui()

    def update_edit_mode_ui(self):
        self.shift_left_btn.setEnabled(self.edit_mode)
        self.shift_right_btn.setEnabled(self.edit_mode)
        if self.edit_mode:
            self.edit_btn.setStyleSheet("background-color: lightblue")
        else:
            self.edit_btn.setStyleSheet("")

    def load_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', 'Videos (*.avi *.tiff *.mp4)')
        if fname:
            self.video_path = fname
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setMaximum(self.total_frames - 1)
            self.show_frame(self.current_frame)
            self.update_motion_plot()

    def load_classifications(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Classifications', '', 'JSON (*.json)')
        if fname:
            with open(fname, 'r') as f:
                self.classified_events = json.load(f)

            # Copy for editing
            self.curated_events = {k: [list(pair) for pair in v] for k, v in self.classified_events.items()}
            all_onsets = [pair[0] for v in self.curated_events.values() for pair in v]
            self.onsets = sorted(all_onsets)
            print("Loaded onsets:", self.onsets)
            self.goto_onset(0)

    def load_motion_energy(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Motion Energy', '', 'CSV/Excel (*.csv *.xlsx)')
        if fname.endswith('.csv'):
            df = pd.read_csv(fname)
        else:
            df = pd.read_excel(fname)
        self.motion_energy = df.select_dtypes(include=[np.number]).iloc[:, 0].values
        self.update_motion_plot()

    def update_motion_plot(self):
        self.ax.clear()
        if self.motion_energy is not None:
            self.ax.plot(self.motion_energy, color='black', linewidth=0.8)
            for onset in self.onsets:
                self.ax.axvline(onset, color='green', linestyle='--', alpha=0.5)
            current_onset = self.onsets[self.current_onset_idx] if self.onsets else self.current_frame
            self.ax.axvline(current_onset, color='red', linewidth=1.5)
            self.ax.set_xlim(0, len(self.motion_energy))
            self.ax.set_ylim(np.min(self.motion_energy), np.max(self.motion_energy))
            self.ax.set_ylabel("Motion")
        self.canvas.draw()

    def play(self):
        self.timer.start(30)

    def pause(self):
        self.timer.stop()

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.slider.setValue(self.current_frame)
            self.show_frame(self.current_frame)

    def show_frame(self, frame_num):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
        self.update_motion_plot()

    def slider_moved(self, value):
        self.current_frame = value
        self.show_frame(value)

    def goto_onset(self, idx):
        if 0 <= idx < len(self.onsets):
            self.current_onset_idx = idx
            self.current_frame = self.onsets[idx]
            self.slider.setValue(self.current_frame)
            self.show_frame(self.current_frame)

    def prev_onset(self):
        if self.current_onset_idx > 0:
            self.goto_onset(self.current_onset_idx - 1)

    def next_onset(self):
        if self.current_onset_idx < len(self.onsets) - 1:
            self.goto_onset(self.current_onset_idx + 1)

    def shift_onset(self, shift):
        if not self.edit_mode:
            return

        current_val = self.onsets[self.current_onset_idx]
        new_val = max(0, min(self.total_frames - 1, current_val + shift))

        # Update onset in data structure
        for cls, segments in self.curated_events.items():
            for pair in segments:
                if pair[0] == current_val:
                    pair[0] = new_val

        self.onsets[self.current_onset_idx] = new_val
        self.onsets.sort()
        self.goto_onset(self.onsets.index(new_val))

    def save_curated_onsets(self):
        with open("curated_classifications.json", 'w') as f:
            json.dump(self.curated_events, f, indent=2)
        print("Saved curated onsets.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    annotator = MotionAnnotator()
    annotator.resize(1000, 750)
    annotator.show()
    sys.exit(app.exec_())
