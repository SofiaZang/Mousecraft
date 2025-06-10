import sys
import cv2
import json
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLabel, QSlider, QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class MotionAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Annotator")
        self.video_path = None
        self.annotations = {}
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
        self.save_btn = QPushButton("Save Annotations")

        self.true_btn.clicked.connect(lambda: self.annotate("true"))
        self.false_btn.clicked.connect(lambda: self.annotate("false"))
        self.edit_btn.clicked.connect(lambda: self.annotate("edit"))

        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.load_video_btn.clicked.connect(self.load_video)
        self.load_anno_btn.clicked.connect(self.load_annotations)
        self.save_btn.clicked.connect(self.save_annotations)

        button_layout = QHBoxLayout()
        for btn in [self.true_btn, self.false_btn, self.edit_btn]:
            button_layout.addWidget(btn)

        nav_layout = QHBoxLayout()
        for btn in [self.play_btn, self.pause_btn, self.load_video_btn, self.load_anno_btn, self.save_btn]:
            nav_layout.addWidget(btn)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
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
        # Save to Excel
        df = pd.DataFrame([
            {'frame': int(k), 'label': v} for k, v in self.annotations.items()
        ])
        df.to_excel("curated_annotations.xlsx", index=False)

        # Save to JSON
        with open("curated_annotations.json", 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print("Annotations saved.")

    def play(self):
        self.timer.start(30)  # ~30 fps

    def pause(self):
        self.timer.stop()

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.slider.setValue(self.current_frame)
            self.show_frame(self.current_frame)

    def show_frame(self, frame_num):
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

    def annotate(self, label):
        self.annotations[self.current_frame] = label
        print(f"Frame {self.current_frame} labeled as {label}")

    def slider_moved(self, value):
        self.current_frame = value
        self.show_frame(value)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    annotator = MotionAnnotator()
    annotator.resize(800, 600)
    annotator.show()
    sys.exit(app.exec_())
