import sys
import cv2
import json
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLabel, QSlider, QFileDialog, QHBoxLayout, QSpinBox, QLabel

)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MotionAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Annotator")
        self.video_path = None
        self.annotations = {}
        self.motion_energy = None
        self.current_frame = 0

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def init_ui(self):
        self.image_label = QLabel("Video will appear here")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)

        self.true_btn = QPushButton("True ✅")
        self.false_btn = QPushButton("False ❌")
        self.edit_btn = QPushButton("Edit ✏️")

        self.play_btn = QPushButton("Play ▶️")
        self.pause_btn = QPushButton("Pause ⏸")
        self.load_video_btn = QPushButton("Load Video")
        self.load_anno_btn = QPushButton("Load Annotations")
        self.load_me_btn = QPushButton("Load Motion Energy")
        self.save_btn = QPushButton("Save Annotations")

        self.true_btn.clicked.connect(lambda: self.annotate("true")) # add buttons 
        self.false_btn.clicked.connect(lambda: self.annotate("false"))
        self.edit_btn.clicked.connect(lambda: self.annotate("edit"))

        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.load_video_btn.clicked.connect(self.load_video)
        self.load_anno_btn.clicked.connect(self.load_annotations)
        self.load_me_btn.clicked.connect(self.load_motion_energy)
        self.save_btn.clicked.connect(self.save_annotations)

        # Layout for annotation buttons
        button_layout = QHBoxLayout()
        for btn in [self.true_btn, self.false_btn, self.edit_btn]:
            button_layout.addWidget(btn)

        # Layout for nav buttons
        nav_layout = QHBoxLayout()
        for btn in [self.play_btn, self.pause_btn, self.load_video_btn,
                    self.load_anno_btn, self.load_me_btn, self.save_btn]:
            nav_layout.addWidget(btn)

        self.speed_label = QLabel("Playback Speed (ms per frame):")
        self.speed_box = QSpinBox()
        self.speed_box.setRange(5, 1000)  # from 5ms to 1000ms per frame
        self.speed_box.setValue(30)       # default 30ms per frame (≈33 FPS)
        self.speed_box.setSuffix(" ms")
        self.speed_box.valueChanged.connect(self.update_timer_interval)

        # Add to nav layout (or a separate layout if you want spacing)
        nav_layout.addWidget(self.speed_label)
        nav_layout.addWidget(self.speed_box)

        # Matplotlib plot for motion energy
        self.fig = Figure(figsize=(5, 1.5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider)
        layout.addLayout(button_layout)
        layout.addLayout(nav_layout)

        self.setLayout(layout)

    def load_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', 'Videos (*.avi *.tiff *.mp4)')
        if fname:
            self.video_path = fname
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setMaximum(self.total_frames - 1)
            self.show_frame(self.current_frame)
            self.update_motion_plot()

    def update_timer_interval(self):
        interval = self.speed_box.value()
        self.timer.setInterval(interval)

    def load_annotations(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Annotations', '', 'Excel/JSON (*.xlsx *.json)')
        if fname.endswith(".json"):
            with open(fname, 'r') as f:
                self.annotations = json.load(f)
        elif fname.endswith(".xlsx"):
            df = pd.read_excel(fname)
            self.annotations = {int(row['frame']): row['label'] for _, row in df.iterrows()}
        print("Loaded annotations:", self.annotations)

    def save_annotations(self):
        df = pd.DataFrame([{'frame': int(k), 'label': v} for k, v in self.annotations.items()])
        df.to_excel("curated_annotations.xlsx", index=False)
        with open("curated_annotations.json", 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print("Annotations saved.")

    def load_motion_energy(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Motion Energy', '', 'CSV/Excel (*.csv *.xlsx)')
        if fname.endswith('.csv'):
            df = pd.read_csv(fname)
        else:
            df = pd.read_excel(fname)

        # Use the first numeric column as motion energy
        self.motion_energy = df.select_dtypes(include=[np.number]).iloc[:, 0].values
        self.update_motion_plot()

    def update_motion_plot(self):
        self.ax.clear()
        if self.motion_energy is not None:
            self.ax.plot(self.motion_energy, color='black', linewidth=0.8)
            self.ax.axvline(self.current_frame, color='red')
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

    def annotate(self, label):
        self.annotations[self.current_frame] = label
        print(f"Frame {self.current_frame} labeled as {label}")

    def slider_moved(self, value):
        self.current_frame = value
        self.show_frame(value)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    annotator = MotionAnnotator()
    annotator.resize(900, 700)
    annotator.show()
    sys.exit(app.exec_())
