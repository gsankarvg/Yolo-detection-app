# main.py

import sys
import cv2
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QLabel, QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from detector import ObjectDetector


class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Video Detection")
        self.setGeometry(100, 100, 800, 600)

        # Detector
        self.detector = ObjectDetector()

        # UI Elements
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        self.open_btn = QPushButton("Open Video")
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")

        self.status_label = QLabel("Status: Ready")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.open_btn)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Video
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # State
        self.is_playing = False
        self.prev_time = 0

        # Button actions
        self.open_btn.clicked.connect(self.open_file)
        self.play_btn.clicked.connect(self.play_video)
        self.pause_btn.clicked.connect(self.pause_video)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi)"
        )

        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.status_label.setText("Video loaded")

    def play_video(self):
        if self.cap:
            self.is_playing = True
            self.timer.start(30)

    def pause_video(self):
        self.is_playing = False
        self.timer.stop()

    def update_frame(self):
        if not self.is_playing or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.status_label.setText("Video finished")
            return

        # Resize (important for CPU)
        frame = cv2.resize(frame, (640, 480))

        # YOLO detection
        processed_frame, vehicle_count = self.detector.detect(frame)

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
        self.prev_time = current_time

        self.status_label.setText(f"Vehicles: {vehicle_count} | FPS: {fps:.2f}")

        # Convert to Qt format
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )

        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())