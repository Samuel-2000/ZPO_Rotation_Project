import sys
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSlider,
                             QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QFileDialog, QComboBox, QCheckBox,
                             QTabWidget, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator, QPainter, QPen
from PIL import Image
import cpp_rotator.rotator_cpp as rotator

class SelectableLabel(QLabel):
    """QLabel, ktorý umožňuje výber obdĺžnikovej oblasti a vykresľuje ho."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_rect = None
        self.rubber_band = None
        self.origin = QPoint()
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if self.pixmap() is None:
            return
        self.origin = event.pos()
        if self.rubber_band is None:
            from PyQt5.QtWidgets import QRubberBand
            self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.rubber_band.setGeometry(QRect(self.origin, self.origin))
        self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if self.rubber_band and self.rubber_band.isVisible():
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if self.rubber_band is None or not self.rubber_band.isVisible():
            return
        self.rubber_band.hide()
        rect = self.rubber_band.geometry()
        if rect.width() < 5 or rect.height() < 5:
            self.selection_rect = None
        else:
            # Prevod na pixmap súradnice (label môže mať padding)
            pixmap = self.pixmap()
            if pixmap is None:
                return
            label_size = self.size()
            pix_size = pixmap.size()
            x_offset = (label_size.width() - pix_size.width()) // 2
            y_offset = (label_size.height() - pix_size.height()) // 2
            pix_rect = rect.translated(-x_offset, -y_offset)
            pix_rect = pix_rect.intersected(QRect(0, 0, pix_size.width(), pix_size.height()))
            if pix_rect.width() <= 0 or pix_rect.height() <= 0:
                self.selection_rect = None
            else:
                self.selection_rect = pix_rect
        self.update()  # prekreslí obdĺžnik
        # Vyšleme signál, že sa zmenil výber (voláme metódu rodiča)
        if hasattr(self.parent(), 'on_selection_changed'):
            self.parent().on_selection_changed()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_rect is not None:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            # Vykreslíme obdĺžnik v súradniciach labelu (treba previesť)
            pixmap = self.pixmap()
            if pixmap:
                label_size = self.size()
                pix_size = pixmap.size()
                x_offset = (label_size.width() - pix_size.width()) // 2
                y_offset = (label_size.height() - pix_size.height()) // 2
                rect_on_label = self.selection_rect.translated(x_offset, y_offset)
                painter.drawRect(rect_on_label)

class RotateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Rotator")
        self.original_image = None
        self.current_image = None
        self.current_angle = 0
        self.cut_corners = True          # predvolene cut
        self.a_value = 4
        self.selection_rect = None       # v pixmap súradniciach (0..pixmap size)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.tabs)

        # Tab 1: Rotátor
        rotator_tab = QWidget()
        self.tabs.addTab(rotator_tab, "Rotator")
        self.init_rotator_tab(rotator_tab)

        # Tab 2: Porovnanie
        comparison_tab = QWidget()
        self.tabs.addTab(comparison_tab, "Comparison")
        self.init_comparison_tab(comparison_tab)

    # ========== Tab 1: Rotátor ==========
    def init_rotator_tab(self, parent):
        layout = QHBoxLayout(parent)

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

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid black")
        layout.addLayout(left, 1)
        layout.addWidget(self.image_label, 4)

        self.update_a_visibility()

    def update_a_visibility(self):
        visible = (self.method == 'lanczos_manual')
        self.a_label.setVisible(visible)
        self.a_slider.setVisible(visible)
        self.a_value_label.setVisible(visible)

    # ========== Tab 2: Porovnanie ==========
    def init_comparison_tab(self, parent):
        # Horizontálny layout: vľavo ovládanie + originál, vpravo mriežka
        main_layout = QHBoxLayout(parent)

        # Ľavý panel (vertikálny)
        left_panel = QVBoxLayout()

        # Originálny obrázok (SelectableLabel)
        self.orig_label = SelectableLabel()
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setMinimumSize(300, 200)
        self.orig_label.setStyleSheet("border: 1px solid black")
        left_panel.addWidget(self.orig_label)

        # Ovládanie uhla
        angle_group = QGroupBox("Angle")
        angle_layout = QVBoxLayout(angle_group)
        self.comp_slider = QSlider(Qt.Horizontal)
        self.comp_slider.setMinimum(-180)
        self.comp_slider.setMaximum(180)
        self.comp_slider.setValue(self.current_angle)
        self.comp_slider.valueChanged.connect(self.on_comp_angle_change)
        angle_layout.addWidget(self.comp_slider)

        self.comp_angle_edit = QLineEdit(str(self.current_angle))
        self.comp_angle_edit.setValidator(QDoubleValidator())
        self.comp_angle_edit.returnPressed.connect(self.on_comp_angle_edit)
        angle_layout.addWidget(self.comp_angle_edit)
        left_panel.addWidget(angle_group)

        # Zoom to content checkbox
        self.zoom_checkbox = QCheckBox("Zoom to content (crop black borders)")
        self.zoom_checkbox.setChecked(True)  # predvolene zapnuté
        self.zoom_checkbox.stateChanged.connect(self.on_zoom_change)
        left_panel.addWidget(self.zoom_checkbox)

        left_panel.addStretch()

        # Pravý panel – mriežka 3x2
        right_panel = QVBoxLayout()
        grid_group = QGroupBox("Double-transformed (rotate then inverse)")
        grid_layout = QGridLayout(grid_group)

        self.method_cells = []  # (name, img_label, psnr_label)

        method_names = [
            'nearest_ref', 'bilinear_ref', 'lanczos_ref',
            'nearest_manual', 'bilinear_manual', 'lanczos_manual'
        ]

        for row in range(3):
            for col in range(2):
                idx = row * 2 + col
                if idx >= len(method_names):
                    break
                name = method_names[idx]

                cell = QWidget()
                cell_layout = QVBoxLayout(cell)

                lbl_name = QLabel(name)
                lbl_name.setAlignment(Qt.AlignCenter)
                cell_layout.addWidget(lbl_name)

                img_label = QLabel()
                img_label.setAlignment(Qt.AlignCenter)
                img_label.setMinimumSize(120, 120)
                img_label.setStyleSheet("border: 1px solid gray")
                cell_layout.addWidget(img_label)

                psnr_label = QLabel("PSNR: -- dB")
                psnr_label.setAlignment(Qt.AlignCenter)
                cell_layout.addWidget(psnr_label)

                grid_layout.addWidget(cell, row, col)
                self.method_cells.append((name, img_label, psnr_label))

        right_panel.addWidget(grid_group)

        # Vložíme ľavý a pravý panel do hlavného layoutu
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)

    # ========== Spracovanie výberu oblasti ==========
    def on_selection_changed(self):
        """Volá sa po zmene výberu v SelectableLabel."""
        pix_rect = self.orig_label.selection_rect
        if pix_rect is None:
            self.selection_rect = None
        else:
            # Prevod na súradnice originálneho obrázka
            pixmap = self.orig_label.pixmap()
            if pixmap is None:
                self.selection_rect = None
                return
            scale_x = self.original_image.shape[1] / pixmap.width()
            scale_y = self.original_image.shape[0] / pixmap.height()
            x = int(pix_rect.x() * scale_x)
            y = int(pix_rect.y() * scale_y)
            w = int(pix_rect.width() * scale_x)
            h = int(pix_rect.height() * scale_y)
            self.selection_rect = (x, y, w, h)
        self.update_comparison()

    # ========== Aktualizácia porovnania ==========
    def update_comparison(self):
        if self.original_image is None:
            return

        # Získame oblasť záujmu
        if self.selection_rect:
            x, y, w, h = self.selection_rect
            # Orežeme na hranice obrázka
            x = max(0, min(x, self.original_image.shape[1]-1))
            y = max(0, min(y, self.original_image.shape[0]-1))
            w = min(w, self.original_image.shape[1] - x)
            h = min(h, self.original_image.shape[0] - y)
            region = self.original_image[y:y+h, x:x+w]
        else:
            region = self.original_image

        # Zobraziť originál (s výberom)
        self.show_image_on_label(self.orig_label, region, max_size=300, keep_selection=True)

        angle = self.current_angle
        a = self.a_value
        cut = self.cut_corners
        zoom = self.zoom_checkbox.isChecked()

        method_funcs = {
            'nearest_ref': rotator.rotate_nearest_ref,
            'bilinear_ref': rotator.rotate_bilinear_ref,
            'lanczos_ref': rotator.rotate_lanczos_ref,
            'nearest_manual': rotator.rotate_nearest_manual,
            'bilinear_manual': rotator.rotate_bilinear_manual,
            'lanczos_manual': lambda img, ang, cut: rotator.rotate_lanczos_manual(img, ang, cut, a)
        }

        for name, img_label, psnr_label in self.method_cells:
            func = method_funcs[name]
            try:
                rotated = func(region, angle, cut)
                inversed = func(rotated, -angle, cut)

                # PSNR na prekrývajúcej sa oblasti
                min_h = min(region.shape[0], inversed.shape[0])
                min_w = min(region.shape[1], inversed.shape[1])
                if cut:
                    # Pri cut sú obe rovnako veľké, ale istota
                    orig_crop = region[:min_h, :min_w]
                    inv_crop = inversed[:min_h, :min_w]
                else:
                    # Pri preserve je inversed väčšia, vyrežeme stred
                    h_orig, w_orig = region.shape[:2]
                    h_inv, w_inv = inversed.shape[:2]
                    start_y = (h_inv - h_orig) // 2
                    start_x = (w_inv - w_orig) // 2
                    if start_y < 0 or start_x < 0:
                        orig_crop = region[:min_h, :min_w]
                        inv_crop = inversed[:min_h, :min_w]
                    else:
                        inv_crop = inversed[start_y:start_y+h_orig, start_x:start_x+w_orig]
                        orig_crop = region

                psnr_val = rotator.psnr(orig_crop, inv_crop)
                psnr_label.setText(f"PSNR: {psnr_val:.2f} dB")

                # Zobrazenie – ak je zoom, ukážeme inv_crop (už orezaný na originál), inak celý inversed
                if zoom and not cut:
                    # inv_crop je orezaný na veľkosť originálu (žiadne čierne okraje)
                    display_img = inv_crop
                else:
                    display_img = inversed
                self.show_image_on_label(img_label, display_img, max_size=120)

            except Exception as e:
                print(f"Chyba pri {name}: {e}")
                psnr_label.setText("PSNR: error")

    def show_image_on_label(self, label, img_np, max_size=120, keep_selection=False):
        """Bezpečné zobrazenie numpy poľa v QLabel. Ak keep_selection=True, zachová sa selection_rect v labeli."""
        if img_np is None or img_np.size == 0:
            return
        img_np = np.ascontiguousarray(img_np)
        h, w, ch = img_np.shape
        bytes_per_line = ch * w
        qimg = QImage(img_np.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        if keep_selection and hasattr(label, 'selection_rect'):
            # Ak ide o SelectableLabel, ponecháme existujúci selection_rect (prepočíta sa v paintEvent)
            label.update()

    # ========== Spoločné funkcie ==========
    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if fname:
            img = Image.open(fname).convert('RGB')
            self.original_image = np.array(img)
            self.current_angle = 0
            self.slider.setValue(0)
            self.angle_edit.setText("0")
            self.comp_slider.setValue(0)
            self.comp_angle_edit.setText("0")
            self.selection_rect = None
            self.orig_label.selection_rect = None
            self.update_image()
            if self.tabs.currentIndex() == 1:
                self.update_comparison()

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
            if self.tabs.currentIndex() == 1:
                self.update_comparison()

    def on_a_change(self, value):
        self.a_value = value
        self.a_value_label.setText(f"a = {value} (window = {2*value})")
        if self.original_image is not None and self.method == 'lanczos_manual':
            self.update_image()
        if self.tabs.currentIndex() == 1:
            self.update_comparison()

    def on_slider_change(self, value):
        self.current_angle = value
        self.angle_edit.setText(str(value))
        if self.original_image is not None:
            self.update_image()
        if self.tabs.currentIndex() == 1:
            self.comp_slider.setValue(value)
            self.comp_angle_edit.setText(str(value))
            self.update_comparison()

    def on_angle_edit(self):
        try:
            angle = float(self.angle_edit.text())
            angle = max(-180, min(180, angle))
            self.current_angle = angle
            self.slider.setValue(int(angle))
            if self.original_image is not None:
                self.update_image()
            if self.tabs.currentIndex() == 1:
                self.comp_slider.setValue(angle)
                self.comp_angle_edit.setText(str(angle))
                self.update_comparison()
        except:
            pass

    def on_comp_angle_change(self, value):
        self.current_angle = value
        self.comp_angle_edit.setText(str(value))
        self.slider.setValue(value)
        self.angle_edit.setText(str(value))
        if self.original_image is not None:
            self.update_comparison()

    def on_comp_angle_edit(self):
        try:
            angle = float(self.comp_angle_edit.text())
            angle = max(-180, min(180, angle))
            self.current_angle = angle
            self.comp_slider.setValue(int(angle))
            self.slider.setValue(int(angle))
            self.angle_edit.setText(str(angle))
            if self.original_image is not None:
                self.update_comparison()
        except:
            pass

    def on_zoom_change(self, state):
        if self.original_image is not None:
            self.update_comparison()

    def on_tab_changed(self, index):
        if index == 1 and self.original_image is not None:
            self.update_comparison()

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
            return

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        self.time_label.setText(f"Processing time: {elapsed_ms:.2f} ms")

        self.current_image = rotated
        self.show_image_on_label(self.image_label, rotated, max_size=400)

    def resizeEvent(self, event):
        if self.current_image is not None:
            self.show_image_on_label(self.image_label, self.current_image, max_size=self.image_label.width())
        super().resizeEvent(event)