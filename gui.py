import sys
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSlider,
                             QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QFileDialog, QComboBox, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator
from PIL import Image
import cpp_rotator.rotator_cpp as rotator

class RotateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Rotator")
        self.original_image = None
        self.current_image = None
        self.current_angle = 0
        self.cut_corners = True
        self.a_value = 4
        self.initUI()

    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()

        # Ľavý panel
        left = QVBoxLayout()

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        left.addWidget(self.load_btn)

        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        left.addWidget(self.save_btn)

        left.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            'nearest_ref', 'bilinear_ref', 'lanczos_ref',
            'nearest_manual', 'bilinear_manual', 'lanczos_manual'
        ])
        self.method_combo.currentTextChanged.connect(self.on_method_change)
        self.method = self.method_combo.currentText()
        left.addWidget(self.method_combo)

        self.cut_checkbox = QCheckBox("Cut corners")
        self.cut_checkbox.setChecked(True)
        self.cut_checkbox.stateChanged.connect(self.on_cut_change)
        left.addWidget(self.cut_checkbox)

        # Ovládanie pre Lanczos parameter a (spočiatku skryté)
        self.a_label = QLabel("Lanczos a (kernel size):")
        self.a_label.setVisible(False)
        left.addWidget(self.a_label)

        self.a_slider = QSlider(Qt.Horizontal)
        self.a_slider.setMinimum(1)
        self.a_slider.setMaximum(6)
        self.a_slider.setValue(self.a_value)
        self.a_slider.setTickInterval(1)
        self.a_slider.setTickPosition(QSlider.TicksBelow)
        self.a_slider.valueChanged.connect(self.on_a_change)
        self.a_slider.setVisible(False)
        left.addWidget(self.a_slider)

        self.a_value_label = QLabel(f"a = {self.a_value} (window = {2*self.a_value})")
        self.a_value_label.setVisible(False)
        left.addWidget(self.a_value_label)

        left.addWidget(QLabel("Angle:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(-180)
        self.slider.setMaximum(180)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_change)
        left.addWidget(self.slider)

        self.angle_edit = QLineEdit("0")
        self.angle_edit.setValidator(QDoubleValidator())
        self.angle_edit.returnPressed.connect(self.on_angle_edit)
        left.addWidget(self.angle_edit)

        self.time_label = QLabel("Processing time: -- ms")
        left.addWidget(self.time_label)

        left.addStretch()

        # Pravý panel – obrázok
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid black")

        main_layout.addLayout(left, 1)
        main_layout.addWidget(self.image_label, 4)
        central.setLayout(main_layout)

        # Počiatočné nastavenie viditeľnosti a-ovládacích prvkov
        self.update_a_visibility()

    def update_a_visibility(self):
        """Zobrazí alebo skryje ovládanie pre parameter a podľa aktuálnej metódy."""
        visible = (self.method == 'lanczos_manual')
        self.a_label.setVisible(visible)
        self.a_slider.setVisible(visible)
        self.a_value_label.setVisible(visible)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if fname:
            img = Image.open(fname).convert('RGB')
            self.original_image = np.array(img)
            self.current_angle = 0
            self.slider.setValue(0)
            self.angle_edit.setText("0")
            self.update_image()

    def save_image(self):
        if self.current_image is not None:
            fname, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg)")
            if fname:
                Image.fromarray(self.current_image).save(fname)

    def on_method_change(self, text):
        self.method = text
        self.update_a_visibility()
        if self.original_image is not None:
            self.update_image()

    def on_cut_change(self, state):
        self.cut_corners = (state == Qt.Checked)
        if self.original_image is not None:
            self.update_image()

    def on_a_change(self, value):
        self.a_value = value
        self.a_value_label.setText(f"a = {value} (window = {2*value})")
        # Aktualizujeme len ak je metóda lanczos_manual
        if self.original_image is not None and self.method == 'lanczos_manual':
            self.update_image()

    def on_slider_change(self, value):
        self.current_angle = value
        self.angle_edit.setText(str(value))
        if self.original_image is not None:
            self.update_image()

    def on_angle_edit(self):
        try:
            angle = float(self.angle_edit.text())
            angle = max(-180, min(180, angle))
            self.current_angle = angle
            self.slider.setValue(int(angle))
            if self.original_image is not None:
                self.update_image()
        except:
            pass

    def update_image(self):
        if self.original_image is None:
            return

        start_time = time.perf_counter()

        try:
            if self.method == 'nearest_ref':
                rotated = rotator.rotate_nearest_ref(self.original_image, self.current_angle, self.cut_corners)
            elif self.method == 'bilinear_ref':
                rotated = rotator.rotate_bilinear_ref(self.original_image, self.current_angle, self.cut_corners)
            elif self.method == 'lanczos_ref':
                rotated = rotator.rotate_lanczos_ref(self.original_image, self.current_angle, self.cut_corners)
            elif self.method == 'nearest_manual':
                rotated = rotator.rotate_nearest_manual(self.original_image, self.current_angle, self.cut_corners)
            elif self.method == 'bilinear_manual':
                rotated = rotator.rotate_bilinear_manual(self.original_image, self.current_angle, self.cut_corners)
            elif self.method == 'lanczos_manual':
                rotated = rotator.rotate_lanczos_manual(self.original_image, self.current_angle, self.cut_corners, self.a_value)
            else:
                return
        except Exception as e:
            print(f"Chyba pri rotácii: {e}")
            import traceback
            traceback.print_exc()
            return

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        self.time_label.setText(f"Processing time: {elapsed_ms:.2f} ms")

        self.current_image = rotated
        h, w, ch = rotated.shape
        bytes_per_line = ch * w
        qimg = QImage(rotated.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        if self.current_image is not None:
            self.update_image()
        super().resizeEvent(event)