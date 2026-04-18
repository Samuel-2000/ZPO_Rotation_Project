# gui.py
import sys
import time
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSlider,
                             QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QFileDialog, QComboBox, QGroupBox,
                             QGridLayout, QRubberBand, QDialog, QProgressDialog,
                             QCheckBox)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator, QPainter, QPen
from PIL import Image
import cpp_rotator.rotator_cpp as rotator

# Matplotlib imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker


class PlotDialog(QDialog):
    """Simple dialog to display a matplotlib figure."""
    def __init__(self, figure, title="PSNR Plot", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        self.resize(800, 600)


class SelectableLabel(QLabel):
    """
    QLabel allowing rectangular selection.
    Stores display_region (x,y,w,h) in original image coordinates
    that corresponds to the currently displayed image.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_rect = None                # QRect in pixmap coordinates (user-drawn)
        self.overlay_selection_rect = None        # QRect in pixmap coordinates (externally set)
        self.display_region = None                 # (x, y, w, h) in original image coordinates
        self.rubber_band = None
        self.origin = QPoint()
        self.setMouseTracking(True)
        self.selection_callback = None

    def set_display_region(self, region):
        """Store the region of the original image that is being displayed."""
        self.display_region = region

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
            pm = self.pixmap()
            if pm is None:
                return
            label_size = self.size()
            pix_size = pm.size()
            x_offset = (label_size.width() - pix_size.width()) // 2
            y_offset = (label_size.height() - pix_size.height()) // 2
            pix_rect = rect.translated(-x_offset, -y_offset)
            pix_rect = pix_rect.intersected(QRect(0, 0, pix_size.width(), pix_size.height()))
            if pix_rect.width() <= 0 or pix_rect.height() <= 0:
                self.selection_rect = None
            else:
                self.selection_rect = pix_rect
        self.update()
        if callable(self.selection_callback):
            try:
                self.selection_callback()
            except Exception:
                pass

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = self.overlay_selection_rect if self.overlay_selection_rect is not None else self.selection_rect
        if rect is not None:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            pm = self.pixmap()
            if pm:
                label_size = self.size()
                pix_size = pm.size()
                x_offset = (label_size.width() - pix_size.width()) // 2
                y_offset = (label_size.height() - pix_size.height()) // 2
                rect_on_label = rect.translated(x_offset, y_offset)
                painter.drawRect(rect_on_label)


class SplitImageLabel(QLabel):
    """
    Split preview widget: left vs right numpy images (HxWx3 uint8).
    Builds a combined pixmap and sets it on the label.
    Selection can be drawn on the entire widget (split line is ignored for overlays).
    """
    selection_changed = None  # callback

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_left = None
        self.img_right = None
        self.split_pos = 50  # percentage
        self._left_ref = None
        self._right_ref = None
        self.selection_rect = None        # QRect in pixmap coordinates (user-drawn)
        self.overlay_selection_rect = None  # QRect in pixmap coordinates (externally set, left side only)
        self.display_region = None         # (x, y, w, h) in original image coords for left image
        self.rubber_band = None
        self.origin = QPoint()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setMouseTracking(True)

    def set_display_region(self, region):
        """Store the region of the original image displayed on the left side."""
        self.display_region = region

    def set_images(self, left, right):
        # Accept None; convert to contiguous arrays if present
        def ensure_rgb(img):
            if img is None:
                return None
            img = np.ascontiguousarray(img)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=2)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            elif img.ndim != 3 or img.shape[2] != 3:
                return None
            return img

        self.img_left = ensure_rgb(left)
        self.img_right = ensure_rgb(right)
        self._left_ref = self.img_left
        self._right_ref = self.img_right
        self._build_and_set_pixmap()

    def set_split(self, pos):
        self.split_pos = max(0, min(100, int(pos)))
        self._build_and_set_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._build_and_set_pixmap()

    def mousePressEvent(self, event):
        if self.pixmap() is None or self.img_left is None:
            return
        # Allow selection anywhere on the label (split line is ignored)
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
            pm = self.pixmap()
            if pm is None:
                return
            label_size = self.size()
            pix_size = pm.size()
            x_offset = (label_size.width() - pix_size.width()) // 2
            y_offset = (label_size.height() - pix_size.height()) // 2
            pix_rect = rect.translated(-x_offset, -y_offset)
            pix_rect = pix_rect.intersected(QRect(0, 0, pix_size.width(), pix_size.height()))
            if pix_rect.width() <= 0 or pix_rect.height() <= 0:
                self.selection_rect = None
            else:
                self.selection_rect = pix_rect
        self.update()
        if callable(self.selection_changed):
            try:
                self.selection_changed()
            except Exception:
                pass

    def paintEvent(self, event):
        super().paintEvent(event)
        """
        # Draw user selection or overlay selection
        if self.overlay_selection_rect is not None:
            # Draw on left half and also duplicate on right half
            self._draw_rectangle(self.overlay_selection_rect)
            # Compute rectangle for right half: same dimensions, shifted by split_x
            pm = self.pixmap()
            if pm is not None:
                pix_size = pm.size()
                split_x = int(pix_size.width() * self.split_pos / 100.0)
                if self.overlay_selection_rect.x() < split_x:
                    # Only duplicate if the rect is within left half (it should be)
                    right_rect = QRect(self.overlay_selection_rect.x() + split_x,
                                       self.overlay_selection_rect.y(),
                                       self.overlay_selection_rect.width(),
                                       self.overlay_selection_rect.height())
                    self._draw_rectangle(right_rect)
        elif self.selection_rect is not None:
            self._draw_rectangle(self.selection_rect)
        """

    def _draw_rectangle(self, rect):
        painter = QPainter(self)
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)
        pm = self.pixmap()
        if pm:
            label_size = self.size()
            pix_size = pm.size()
            x_offset = (label_size.width() - pix_size.width()) // 2
            y_offset = (label_size.height() - pix_size.height()) // 2
            rect_on_label = rect.translated(x_offset, y_offset)
            painter.drawRect(rect_on_label)

    def _build_and_set_pixmap(self):
        if self.img_left is None or self.img_right is None:
            self.clear()
            return
        try:
            left = self.img_left
            right = self.img_right

            tgt_w = max(1, self.width())
            tgt_h = max(1, self.height())

            qimg_left = QImage(left.data, left.shape[1], left.shape[0], 3 * left.shape[1], QImage.Format_RGB888)
            qimg_right = QImage(right.data, right.shape[1], right.shape[0], 3 * right.shape[1], QImage.Format_RGB888)

            pix_left = QPixmap.fromImage(qimg_left).scaled(tgt_w, tgt_h, Qt.KeepAspectRatio, Qt.FastTransformation)
            pix_right = QPixmap.fromImage(qimg_right).scaled(tgt_w, tgt_h, Qt.KeepAspectRatio, Qt.FastTransformation)

            canvas_w = max(pix_left.width(), pix_right.width())
            canvas_h = max(pix_left.height(), pix_right.height())
            canvas = QPixmap(canvas_w, canvas_h)
            canvas.fill(Qt.transparent)

            painter = QPainter(canvas)
            painter.drawPixmap(0, 0, pix_left)
            split_x = int(canvas_w * self.split_pos / 100.0)
            painter.setClipRect(split_x, 0, canvas_w - split_x, canvas_h)
            painter.drawPixmap(0, 0, pix_right)
            painter.setClipping(False)
            pen = QPen(Qt.black, 2)
            painter.setPen(pen)
            painter.drawLine(split_x, 0, split_x, canvas_h)
            painter.end()

            final = canvas.scaled(tgt_w, tgt_h, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.setPixmap(final)
            self._last_pixmap = final
            self._left_ref = left
            self._right_ref = right
        except Exception as e:
            print("SplitImageLabel: error building pixmap:", e)
            self.clear()


class RotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Rotator")

        # core state
        self.original_image = None    # HxWx3 uint8
        self.current_image = None
        self.current_angle = 0.0
        self.zoom_mode = 'cut'
        self.a_value = 4
        self.selection_rect = None    # (x,y,w,h) in ORIGINAL image coords

        # split preview currently-selected method
        self.split_method = 'nearest_ref'

        # Show/hide manual methods
        self.show_manual = True

        # Range limiting
        self.limit_range = True

        # Caching for PSNR plots
        self.psnr_cache = {}           # dict {method: [(angle, psnr), ...]}
        self.cache_angles = None        # list of angles used
        self.cache_params = {}           # {'a': a, 'cut': cut_mode}  (selection cleared separately)

        # Mapping of method names to functions (for later use)
        self.method_funcs = {
            'nearest_ref': rotator.rotate_nearest_ref,
            'bilinear_ref': rotator.rotate_bilinear_ref,
            'lanczos_ref': rotator.rotate_lanczos_ref,   # fixed a=4 (OpenCV)
            'nearest_manual': rotator.rotate_nearest_manual,
            'bilinear_manual': rotator.rotate_bilinear_manual,
            'lanczos_manual': lambda img, ang, cut: rotator.rotate_lanczos_manual(img, ang, cut, self.a_value)
        }

        self.initUI()

        # Auto-load lena.png if present
        if os.path.exists("lena.png"):
            self.load_image_file("lena.png")

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        from PyQt5.QtWidgets import QTabWidget
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.tabs)

        # Rotator tab
        rotator_tab = QWidget()
        self.tabs.addTab(rotator_tab, "Rotator")
        self.init_rotator_tab(rotator_tab)

        # Comparison tab
        comparison_tab = QWidget()
        self.tabs.addTab(comparison_tab, "Comparison")
        self.init_comparison_tab(comparison_tab)

    # ---------- Rotator tab ----------
    def init_rotator_tab(self, parent):
        layout = QHBoxLayout(parent)
        left = QVBoxLayout()

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        left.addWidget(self.load_btn)

        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        left.addWidget(self.save_btn)

        self.pattern_btn = QPushButton("Generate Checkerboard (FHD)")
        self.pattern_btn.clicked.connect(self.generate_checkerboard)
        left.addWidget(self.pattern_btn)

        left.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            'nearest_ref', 'bilinear_ref', 'lanczos_ref (a=4 fixed)',
            'nearest_manual', 'bilinear_manual', 'lanczos_manual'
        ])
        self.method_combo.currentTextChanged.connect(self.on_method_change)
        self.method = self.method_combo.currentText()
        left.addWidget(self.method_combo)

        left.addWidget(QLabel("Zoom mode:"))
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(['cut', 'preserve', 'zoom_to_content'])
        self.zoom_combo.setCurrentText(self.zoom_mode)
        self.zoom_combo.currentTextChanged.connect(self.on_zoom_mode_change)
        left.addWidget(self.zoom_combo)

        self.a_label = QLabel("Lanczos a (kernel size, manual only):")
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
        self.image_label.setMinimumSize(500, 420)
        self.image_label.setStyleSheet("border: 1px solid black")
        layout.addLayout(left, 1)
        layout.addWidget(self.image_label, 4)

        self.update_a_visibility()

    def update_a_visibility(self):
        visible = ('lanczos_manual' in self.method)
        self.a_label.setVisible(visible)
        self.a_slider.setVisible(visible)
        self.a_value_label.setVisible(visible)

    # ---------- Comparison tab ----------
    def init_comparison_tab(self, parent):
        main_layout = QHBoxLayout(parent)
        left_panel = QVBoxLayout()

        # Original image (selectable)
        self.orig_label = SelectableLabel()
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setMinimumSize(380, 300)
        self.orig_label.setStyleSheet("border: 1px solid black")
        left_panel.addWidget(self.orig_label)
        self.orig_label.selection_callback = self.on_user_selection_changed

        # Split preview controls
        split_group = QGroupBox("Zoomed split comparison (select on left side)")
        split_layout = QVBoxLayout(split_group)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.split_method_combo = QComboBox()
        self.split_method_combo.addItems([
            'nearest_ref', 'bilinear_ref', 'lanczos_ref (a=4 fixed)',
            'nearest_manual', 'bilinear_manual', 'lanczos_manual'
        ])
        self.split_method_combo.currentTextChanged.connect(self.on_split_method_change)
        method_layout.addWidget(self.split_method_combo)
        split_layout.addLayout(method_layout)

        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.setTickInterval(10)
        self.split_slider.setTickPosition(QSlider.TicksBelow)
        self.split_slider.valueChanged.connect(self.on_split_slider_change)
        split_layout.addWidget(self.split_slider)

        self.split_image_label = SplitImageLabel()
        self.split_image_label.setAlignment(Qt.AlignCenter)
        self.split_image_label.setMinimumSize(260, 180)
        self.split_image_label.setStyleSheet("border: 1px solid #888;")
        self.split_image_label.selection_changed = self.on_split_selection_changed
        split_layout.addWidget(self.split_image_label)

        left_panel.addWidget(split_group)

        angle_group = QGroupBox("Angle")
        angle_layout = QVBoxLayout(angle_group)
        self.comp_slider = QSlider(Qt.Horizontal)
        self.comp_slider.setMinimum(-180)
        self.comp_slider.setMaximum(180)
        self.comp_slider.setValue(int(self.current_angle))
        self.comp_slider.valueChanged.connect(self.on_comp_angle_change)
        angle_layout.addWidget(self.comp_slider)

        self.comp_angle_edit = QLineEdit(str(self.current_angle))
        self.comp_angle_edit.setValidator(QDoubleValidator())
        self.comp_angle_edit.returnPressed.connect(self.on_comp_angle_edit)
        angle_layout.addWidget(self.comp_angle_edit)
        left_panel.addWidget(angle_group)

        # PSNR analysis buttons + checkboxes
        psnr_group = QGroupBox("PSNR Analysis")
        psnr_layout = QVBoxLayout(psnr_group)
        btn_layout = QHBoxLayout()
        self.btn_psnr_line = QPushButton("PSNR vs Angle")
        self.btn_psnr_line.clicked.connect(self.show_psnr_line_plot)
        self.btn_psnr_box = QPushButton("Boxplot of PSNR")
        self.btn_psnr_box.clicked.connect(self.show_psnr_boxplot)
        btn_layout.addWidget(self.btn_psnr_line)
        btn_layout.addWidget(self.btn_psnr_box)
        psnr_layout.addLayout(btn_layout)

        self.cb_show_manual = QCheckBox("Show manual methods")
        self.cb_show_manual.setChecked(self.show_manual)
        self.cb_show_manual.stateChanged.connect(self.on_show_manual_toggled)
        psnr_layout.addWidget(self.cb_show_manual)

        self.cb_limit_range = QCheckBox("Limit angle range (45° square, 90° rectangle)")
        self.cb_limit_range.setChecked(self.limit_range)
        self.cb_limit_range.stateChanged.connect(self.on_limit_range_toggled)
        self.cb_limit_range.setToolTip(
            "When checked, only positive angles up to 90° (or 45° for square images) are computed.\n"
            "This speeds up computation by avoiding symmetric angles."
        )
        psnr_layout.addWidget(self.cb_limit_range)

        left_panel.addWidget(psnr_group)

        left_panel.addWidget(QLabel("Zoom mode:"))
        self.zoom_combo_comp = QComboBox()
        self.zoom_combo_comp.addItems(['cut', 'preserve', 'zoom_to_content'])
        self.zoom_combo_comp.setCurrentText(self.zoom_mode)
        self.zoom_combo_comp.currentTextChanged.connect(self.on_zoom_change)
        left_panel.addWidget(self.zoom_combo_comp)

        lanczos_group = QGroupBox("Lanczos a (manual)")
        lanczos_layout = QVBoxLayout(lanczos_group)
        self.comp_a_slider = QSlider(Qt.Horizontal)
        self.comp_a_slider.setMinimum(1)
        self.comp_a_slider.setMaximum(6)
        self.comp_a_slider.setValue(self.a_value)
        self.comp_a_slider.setTickInterval(1)
        self.comp_a_slider.setTickPosition(QSlider.TicksBelow)
        self.comp_a_slider.valueChanged.connect(self.on_comp_a_change)
        lanczos_layout.addWidget(self.comp_a_slider)

        self.comp_a_label = QLabel(f"a = {self.a_value} (window = {2*self.a_value})")
        lanczos_layout.addWidget(self.comp_a_label)
        left_panel.addWidget(lanczos_group)

        left_panel.addStretch()

        right_panel = QVBoxLayout()
        grid_group = QGroupBox("Double-transformed (ref | manual)")
        grid_layout = QGridLayout(grid_group)

        pairs = [
            ('nearest_ref', 'nearest_manual'),
            ('bilinear_ref', 'bilinear_manual'),
            ('lanczos_ref', 'lanczos_manual'),
        ]
        self.method_cells = []
        self.manual_cells = []   # list of manual column widgets for show/hide
        for row, (ref_name, man_name) in enumerate(pairs):
            ref_cell = QWidget()
            ref_layout = QVBoxLayout(ref_cell)
            if ref_name == 'lanczos_ref':
                lbl_ref = QLabel("lanczos ref (a=4)")
            else:
                lbl_ref = QLabel(ref_name)
            lbl_ref.setAlignment(Qt.AlignCenter)
            ref_layout.addWidget(lbl_ref)
            ref_img = QLabel()
            ref_img.setAlignment(Qt.AlignCenter)
            ref_img.setMinimumSize(260, 240)
            ref_img.setStyleSheet("border: 1px solid gray")
            ref_layout.addWidget(ref_img)

            # PSNR label with plot button for ref
            ref_psnr_widget = QWidget()
            ref_psnr_layout = QHBoxLayout(ref_psnr_widget)
            ref_psnr_layout.setContentsMargins(0,0,0,0)
            ref_psnr = QLabel("PSNR: -- dB")
            ref_psnr_btn = QPushButton("📈")
            ref_psnr_btn.setFixedSize(24, 24)
            ref_psnr_btn.clicked.connect(lambda checked, m=ref_name: self.show_method_line_plot(m))
            ref_psnr_layout.addWidget(ref_psnr)
            ref_psnr_layout.addWidget(ref_psnr_btn)
            ref_layout.addWidget(ref_psnr_widget)

            grid_layout.addWidget(ref_cell, row, 0)

            man_cell = QWidget()
            man_layout = QVBoxLayout(man_cell)
            lbl_man = QLabel(man_name)
            lbl_man.setAlignment(Qt.AlignCenter)
            man_layout.addWidget(lbl_man)
            man_img = QLabel()
            man_img.setAlignment(Qt.AlignCenter)
            man_img.setMinimumSize(260, 240)
            man_img.setStyleSheet("border: 1px solid gray")
            man_layout.addWidget(man_img)

            # PSNR label with plot button for man
            man_psnr_widget = QWidget()
            man_psnr_layout = QHBoxLayout(man_psnr_widget)
            man_psnr_layout.setContentsMargins(0,0,0,0)
            man_psnr = QLabel("PSNR: -- dB")
            man_psnr_btn = QPushButton("📈")
            man_psnr_btn.setFixedSize(24, 24)
            man_psnr_btn.clicked.connect(lambda checked, m=man_name: self.show_method_line_plot(m))
            man_psnr_layout.addWidget(man_psnr)
            man_psnr_layout.addWidget(man_psnr_btn)
            man_layout.addWidget(man_psnr_widget)

            grid_layout.addWidget(man_cell, row, 1)

            self.method_cells.append((ref_name, ref_img, ref_psnr, ref_psnr_btn,
                                      man_name, man_img, man_psnr, man_psnr_btn))
            self.manual_cells.append(man_cell)

        right_panel.addWidget(grid_group)
        right_panel.addStretch()

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)

    # ---------- Show/hide manual methods and range limit ----------
    def on_show_manual_toggled(self, state):
        self.show_manual = (state == Qt.Checked)
        for cell in self.manual_cells:
            cell.setVisible(self.show_manual)
        self.clear_cache()
        if self.original_image is not None:
            self.update_comparison()

    def on_limit_range_toggled(self, state):
        self.limit_range = (state == Qt.Checked)
        self.clear_cache()
        if self.original_image is not None:
            print(f"Angle range limiting {'enabled' if self.limit_range else 'disabled'}")

    # ---------- Selection callbacks ----------
    def on_user_selection_changed(self):
        """Called when user draws on the original image label."""
        if self.original_image is None:
            self.selection_rect = None
            return

        pix_rect = self.orig_label.selection_rect
        if pix_rect is None:
            self.selection_rect = None
        else:
            pm = self.orig_label.pixmap()
            if pm is None:
                self.selection_rect = None
                return
            pix_w = pm.width()
            pix_h = pm.height()
            orig_h, orig_w = self.original_image.shape[:2]
            scale_x = orig_w / pix_w
            scale_y = orig_h / pix_h

            x = int(round(pix_rect.x() * scale_x))
            y = int(round(pix_rect.y() * scale_y))
            w = int(round(pix_rect.width() * scale_x))
            h = int(round(pix_rect.height() * scale_y))

            x = max(0, min(x, orig_w - 1))
            y = max(0, min(y, orig_h - 1))
            w = max(1, min(w, orig_w - x))
            h = max(1, min(h, orig_h - y))

            self.selection_rect = (x, y, w, h)

        self.clear_cache()
        self.update_comparison()

    def on_split_selection_changed(self):
        """Called when user draws on the split image label (left side only)."""
        if self.original_image is None:
            return

        pix_rect = self.split_image_label.selection_rect
        if pix_rect is None:
            self.selection_rect = None
        else:
            display_region = self.split_image_label.display_region
            if display_region is None:
                return

            pm = self.split_image_label.pixmap()
            if pm is None:
                return
            pix_w = pm.width()
            pix_h = pm.height()

            # The left side corresponds to the zoomed original region
            orig_x0, orig_y0, orig_w_disp, orig_h_disp = display_region

            x = orig_x0 + int(round(pix_rect.x() * orig_w_disp / pix_w))
            y = orig_y0 + int(round(pix_rect.y() * orig_h_disp / pix_h))
            w = int(round(pix_rect.width() * orig_w_disp / pix_w))
            h = int(round(pix_rect.height() * orig_h_disp / pix_h))

            orig_h, orig_w = self.original_image.shape[:2]
            x = max(0, min(x, orig_w - 1))
            y = max(0, min(y, orig_h - 1))
            w = max(1, min(w, orig_w - x))
            h = max(1, min(h, orig_h - y))

            self.selection_rect = (x, y, w, h)

        self.clear_cache()
        self.update_comparison()

    # ---------- Helpers ----------
    def _center_crop(self, img, H, W):
        hh, ww = img.shape[:2]
        if H > hh or W > ww:
            H = min(H, hh)
            W = min(W, ww)
        sy = (hh - H) // 2
        sx = (ww - W) // 2
        return img[sy:sy+H, sx:sx+W]

    def _centroid_of_mask(self, mask):
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None
        cx = np.mean(xs)
        cy = np.mean(ys)
        return (cx, cy)

    def _align_by_centroid_and_crop(self, orig, inv):
        if orig is None or inv is None:
            return None, None, 0, 0

        if orig.shape == inv.shape and np.array_equal(orig, inv):
            return orig, inv, 0, 0

        orig_h, orig_w = orig.shape[:2]
        inv_h, inv_w = inv.shape[:2]

        mask_o = np.any(orig != 0, axis=2)
        mask_i = np.any(inv != 0, axis=2)

        cent_o = self._centroid_of_mask(mask_o)
        cent_i = self._centroid_of_mask(mask_i)

        if cent_o is None or cent_i is None:
            common_h = min(orig_h, inv_h)
            common_w = min(orig_w, inv_w)
            orig_crop = self._center_crop(orig, common_h, common_w)
            inv_crop = self._center_crop(inv, common_h, common_w)
            orig_x0 = (orig_w - common_w) // 2
            orig_y0 = (orig_h - common_h) // 2
            return orig_crop, inv_crop, orig_x0, orig_y0

        dx = cent_o[0] - cent_i[0]
        dy = cent_o[1] - cent_i[1]

        if abs(dx) < 1.5 and abs(dy) < 1.5 and orig.shape == inv.shape:
            return orig, inv, 0, 0

        if abs(dx) < 0.5 and abs(dy) < 0.5:
            common_h = min(orig_h, inv_h)
            common_w = min(orig_w, inv_w)
            orig_crop = self._center_crop(orig, common_h, common_w)
            inv_crop = self._center_crop(inv, common_h, common_w)
            orig_x0 = (orig_w - common_w) // 2
            orig_y0 = (orig_h - common_h) // 2
            return orig_crop, inv_crop, orig_x0, orig_y0

        dx = int(round(dx))
        dy = int(round(dy))

        ox0 = max(0, dx)
        ix0 = max(0, -dx)
        ow = min(orig_w - ox0, inv_w - ix0)
        if ow <= 0:
            common_h = min(orig_h, inv_h)
            common_w = min(orig_w, inv_w)
            orig_crop = self._center_crop(orig, common_h, common_w)
            inv_crop = self._center_crop(inv, common_h, common_w)
            orig_x0 = (orig_w - common_w) // 2
            orig_y0 = (orig_h - common_h) // 2
            return orig_crop, inv_crop, orig_x0, orig_y0

        oy0 = max(0, dy)
        iy0 = max(0, -dy)
        oh = min(orig_h - oy0, inv_h - iy0)
        if oh <= 0:
            common_h = min(orig_h, inv_h)
            common_w = min(orig_w, inv_w)
            orig_crop = self._center_crop(orig, common_h, common_w)
            inv_crop = self._center_crop(inv, common_h, common_w)
            orig_x0 = (orig_w - common_w) // 2
            orig_y0 = (orig_h - common_h) // 2
            return orig_crop, inv_crop, orig_x0, orig_y0

        orig_crop = orig[oy0:oy0+oh, ox0:ox0+ow]
        inv_crop = inv[iy0:iy0+oh, ix0:ix0+ow]
        return orig_crop, inv_crop, ox0, oy0

    def _map_selection_to_common_with_origin(self, sel, orig_crop_origin, common_w, common_h):
        if sel is None:
            return None
        x, y, w, h = sel
        ox0, oy0 = orig_crop_origin
        rx0 = x - ox0
        ry0 = y - oy0
        ix0 = max(0, rx0)
        iy0 = max(0, ry0)
        ix1 = min(common_w, rx0 + w)
        iy1 = min(common_h, ry0 + h)
        iw = ix1 - ix0
        ih = iy1 - iy0
        if iw <= 0 or ih <= 0:
            return None
        return (int(ix0), int(iy0), int(iw), int(ih))

    # ---------- Comparison update ----------
    def update_comparison(self):
        if self.original_image is None:
            return

        orig_h, orig_w = self.original_image.shape[:2]

        # Show original with overlay
        self.show_image_on_label(self.orig_label, self.original_image, max_size=380,
                                 selection=self.selection_rect, set_overlay=True,
                                 display_region=(0, 0, orig_w, orig_h))

        # Set original region for split view (zoom left image)
        if self.selection_rect is not None:
            x, y, w, h = self.selection_rect
            self.zoom_original_region = self.original_image[y:y+h, x:x+w].copy()
            zoom_display_region = (x, y, w, h)
        else:
            # No selection: use full original image
            self.zoom_original_region = self.original_image
            zoom_display_region = (0, 0, orig_w, orig_h)

        src = self.original_image
        angle = float(self.current_angle)
        a = int(self.a_value)
        zoom_mode = self.zoom_combo_comp.currentText()
        cut = (zoom_mode == 'cut')
        use_zoom = (zoom_mode == 'zoom_to_content')

        method_funcs = {
            'nearest_ref': rotator.rotate_nearest_ref,
            'bilinear_ref': rotator.rotate_bilinear_ref,
            'lanczos_ref': rotator.rotate_lanczos_ref,
            'nearest_manual': rotator.rotate_nearest_manual,
            'bilinear_manual': rotator.rotate_bilinear_manual,
            'lanczos_manual': lambda img, ang, cut_flag: rotator.rotate_lanczos_manual(img, ang, cut_flag, a)
        }

        # ---------------------------
        # SPLIT PREVIEW: produce twice-transformed (rotate -> inverse) image to match grid
        # ---------------------------
        split_method = self.split_method_combo.currentText()
        if split_method == 'lanczos_ref (a=4 fixed)':
            split_method = 'lanczos_ref'
        right_img = None
        try:
            if split_method in method_funcs:
                fn = method_funcs[split_method]
                rot_img = fn(src, angle, cut)
                inv_img = fn(rot_img, -angle, cut)

                display_inv = inv_img
                if use_zoom:
                    iw, ih = rotator.get_max_inner_rect(src.shape[1], src.shape[0], angle)
                    iw = max(1, int(iw))
                    ih = max(1, int(ih))
                    display_inv = self._center_crop(inv_img, ih, iw)

                orig_crop, inv_crop, orig_x0, orig_y0 = self._align_by_centroid_and_crop(src, display_inv)

                if inv_crop is None:
                    zh, zw = self.zoom_original_region.shape[:2]
                    right_img = np.zeros((zh, zw, 3), dtype=np.uint8)
                else:
                    if self.selection_rect is not None:
                        sel_common = self._map_selection_to_common_with_origin(self.selection_rect, (orig_x0, orig_y0),
                                                                               inv_crop.shape[1], inv_crop.shape[0])
                        if sel_common is not None:
                            x0, y0, w0, h0 = sel_common
                            right_img = inv_crop[y0:y0+h0, x0:x0+w0].copy()
                        else:
                            zh, zw = self.zoom_original_region.shape[:2]
                            right_img = np.zeros((zh, zw, 3), dtype=np.uint8)
                    else:
                        zh, zw = self.zoom_original_region.shape[:2]
                        right_img = self._center_crop(inv_crop, zh, zw)
        except Exception as e:
            print(f"Split preview error for method {split_method}: {e}")
            right_img = None

        # Update split view widget, storing display region for left side
        if right_img is not None and self.zoom_original_region is not None:
            self.split_image_label.set_images(self.zoom_original_region, right_img)
            self.split_image_label.set_split(self.split_slider.value())
            self.split_image_label.set_display_region(zoom_display_region)
            # Update overlay on split image if selection exists
            if self.selection_rect is not None:
                # Map selection to split label coordinates (left side only)
                ox, oy, ow, oh = zoom_display_region
                sx, sy, sw, sh = self.selection_rect
                inter_x0 = max(sx, ox)
                inter_y0 = max(sy, oy)
                inter_x1 = min(sx+sw, ox+ow)
                inter_y1 = min(sy+sh, oy+oh)
                if inter_x0 < inter_x1 and inter_y0 < inter_y1:
                    disp_x = inter_x0 - ox
                    disp_y = inter_y0 - oy
                    disp_w = inter_x1 - inter_x0
                    disp_h = inter_y1 - inter_y0
                    # Scale to label coordinates
                    label_w = self.split_image_label.pixmap().width()
                    label_h = self.split_image_label.pixmap().height()
                    scale_x = label_w / ow
                    scale_y = label_h / oh
                    label_sel_x = int(round(disp_x * scale_x))
                    label_sel_y = int(round(disp_y * scale_y))
                    label_sel_w = max(1, int(round(disp_w * scale_x)))
                    label_sel_h = max(1, int(round(disp_h * scale_y)))
                    self.split_image_label.overlay_selection_rect = QRect(label_sel_x, label_sel_y, label_sel_w, label_sel_h)
                else:
                    self.split_image_label.overlay_selection_rect = None
            else:
                self.split_image_label.overlay_selection_rect = None
            self.split_image_label.update()

        # ---------------------------
        # 3x2 GRID
        # ---------------------------
        for (ref_name, ref_img_label, ref_psnr_label, ref_btn,
             man_name, man_img_label, man_psnr_label, man_btn) in self.method_cells:
            try:
                # Reference method
                ref_fn = method_funcs[ref_name]
                ref_rot = ref_fn(src, angle, cut)
                ref_inv = ref_fn(ref_rot, -angle, cut)

                display_ref = ref_inv
                if use_zoom:
                    iw, ih = rotator.get_max_inner_rect(src.shape[1], src.shape[0], angle)
                    iw = max(1, int(iw))
                    ih = max(1, int(ih))
                    display_ref = self._center_crop(ref_inv, ih, iw)

                orig_crop_ref, inv_crop_ref, orig_x0_ref, orig_y0_ref = self._align_by_centroid_and_crop(src, display_ref)
                sel = self.selection_rect

                sel_ref_in_common = self._map_selection_to_common_with_origin(sel, (orig_x0_ref, orig_y0_ref),
                                                                             inv_crop_ref.shape[1], inv_crop_ref.shape[0]) if (orig_crop_ref is not None and inv_crop_ref is not None) else None

                def psnr_on_region(orig_c, inv_c, sel_common):
                    if orig_c is None or inv_c is None:
                        return -1.0
                    if sel_common is None:
                        return rotator.psnr(orig_c, inv_c)
                    x0, y0, w0, h0 = sel_common
                    if w0 <= 0 or h0 <= 0:
                        return -1.0
                    o_crop = orig_c[y0:y0+h0, x0:x0+w0].copy()   # make contiguous
                    i_crop = inv_c[y0:y0+h0, x0:x0+w0].copy()   # make contiguous
                    return rotator.psnr(o_crop, i_crop)

                psnr_ref = psnr_on_region(orig_crop_ref, inv_crop_ref, sel_ref_in_common)
                ref_psnr_label.setText(f"PSNR: {psnr_ref:.2f} dB" if psnr_ref >= 0 else "PSNR: -- dB")

                def display_for_label(orig_c, inv_c, orig_origin, sel_common, label):
                    if inv_c is None:
                        label.clear()
                        return
                    if sel_common is None:
                        self.show_image_on_label(label, inv_c, max_size=360)
                    else:
                        x0, y0, w0, h0 = sel_common
                        img = inv_c[y0:y0+h0, x0:x0+w0]
                        self.show_image_on_label(label, img, max_size=360)

                display_for_label(orig_crop_ref, inv_crop_ref, (orig_x0_ref, orig_y0_ref), sel_ref_in_common, ref_img_label)

                # Manual method only if enabled
                if self.show_manual:
                    man_fn = method_funcs[man_name]
                    man_rot = man_fn(src, angle, cut)
                    man_inv = man_fn(man_rot, -angle, cut)

                    display_man = man_inv
                    if use_zoom:
                        iw, ih = rotator.get_max_inner_rect(src.shape[1], src.shape[0], angle)
                        iw = max(1, int(iw))
                        ih = max(1, int(ih))
                        display_man = self._center_crop(man_inv, ih, iw)

                    orig_crop_man, inv_crop_man, orig_x0_man, orig_y0_man = self._align_by_centroid_and_crop(src, display_man)
                    sel_man_in_common = self._map_selection_to_common_with_origin(sel, (orig_x0_man, orig_y0_man),
                                                                                 inv_crop_man.shape[1], inv_crop_man.shape[0]) if (orig_crop_man is not None and inv_crop_man is not None) else None

                    psnr_man = psnr_on_region(orig_crop_man, inv_crop_man, sel_man_in_common)
                    man_psnr_label.setText(f"PSNR: {psnr_man:.2f} dB" if psnr_man >= 0 else "PSNR: -- dB")
                    display_for_label(orig_crop_man, inv_crop_man, (orig_x0_man, orig_y0_man), sel_man_in_common, man_img_label)
                else:
                    man_psnr_label.setText("PSNR: -- dB")
                    man_img_label.clear()

            except Exception as e:
                print(f"Error computing {ref_name}/{man_name}: {e}")
                ref_psnr_label.setText("PSNR: error")
                man_psnr_label.setText("PSNR: error")

    # ---------- Show image helper ----------
    def show_image_on_label(self, label, img_np, max_size=120, selection=None, set_overlay=False, display_region=None):
        if img_np is None or img_np.size == 0:
            label.clear()
            return
        img_np = np.ascontiguousarray(img_np)
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        if img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=2)
        elif img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        elif img_np.ndim != 3 or img_np.shape[2] != 3:
            label.clear()
            return

        h, w = img_np.shape[:2]
        bytes_per_line = 3 * w

        qimg = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label._img_ref = img_np

        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        label.setPixmap(scaled)

        if isinstance(label, SelectableLabel) and display_region is not None:
            label.set_display_region(display_region)

        if isinstance(label, SelectableLabel):
            if set_overlay and (selection is not None) and (self.original_image is not None):
                if display_region is not None:
                    orig_x0, orig_y0, orig_w_disp, orig_h_disp = display_region
                    sel_x, sel_y, sel_w, sel_h = selection
                    inter_x0 = max(sel_x, orig_x0)
                    inter_y0 = max(sel_y, orig_y0)
                    inter_x1 = min(sel_x + sel_w, orig_x0 + orig_w_disp)
                    inter_y1 = min(sel_y + sel_h, orig_y0 + orig_h_disp)
                    if inter_x0 < inter_x1 and inter_y0 < inter_y1:
                        disp_sel_x = inter_x0 - orig_x0
                        disp_sel_y = inter_y0 - orig_y0
                        disp_sel_w = inter_x1 - inter_x0
                        disp_sel_h = inter_y1 - inter_y0
                        label_w = scaled.width()
                        label_h = scaled.height()
                        scale_x = label_w / orig_w_disp
                        scale_y = label_h / orig_h_disp
                        label_sel_x = int(round(disp_sel_x * scale_x))
                        label_sel_y = int(round(disp_sel_y * scale_y))
                        label_sel_w = max(1, int(round(disp_sel_w * scale_x)))
                        label_sel_h = max(1, int(round(disp_sel_h * scale_y)))
                        label.overlay_selection_rect = QRect(label_sel_x, label_sel_y, label_sel_w, label_sel_h)
                    else:
                        label.overlay_selection_rect = None
                else:
                    orig_h, orig_w = self.original_image.shape[:2]
                    label_w = scaled.width()
                    label_h = scaled.height()
                    scale_x = label_w / orig_w
                    scale_y = label_h / orig_h
                    sel_x, sel_y, sel_w, sel_h = selection
                    label_sel_x = int(round(sel_x * scale_x))
                    label_sel_y = int(round(sel_y * scale_y))
                    label_sel_w = max(1, int(round(sel_w * scale_x)))
                    label_sel_h = max(1, int(round(sel_h * scale_y)))
                    label.overlay_selection_rect = QRect(label_sel_x, label_sel_y, label_sel_w, label_sel_h)
            else:
                label.overlay_selection_rect = None
            label.update()

    # ---------- Load / Save / Generate ----------
    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if not fname:
            return
        self.load_image_file(fname)

    def load_image_file(self, fname):
        img = Image.open(fname).convert('RGB')
        self.original_image = np.array(img)
        print("Loaded image:", None if self.original_image is None else (self.original_image.shape, self.original_image.dtype))
        self.current_angle = 0.0
        self.slider.setValue(0)
        self.angle_edit.setText("0")
        self.comp_slider.setValue(0)
        self.comp_angle_edit.setText("0")
        self.selection_rect = None
        self.orig_label.selection_rect = None
        self.split_image_label.selection_rect = None
        self.orig_label.overlay_selection_rect = None
        self.split_image_label.overlay_selection_rect = None
        self.clear_cache()
        self.update_image()
        if self.tabs.currentIndex() == 1:
            self.update_comparison()

    def save_image(self):
        if self.current_image is None:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg)")
        if fname:
            Image.fromarray(self.current_image).save(fname)

    def generate_checkerboard(self):
        width, height = 1920, 1080
        colors = np.array([
            [255, 0, 0],       # red
            [0, 255, 0],       # green
            [0, 0, 255],       # blue
            [255, 255, 0],     # yellow
            [0, 255, 255],     # cyan
            [255, 0, 255],     # magenta
            [0, 0, 0],         # black
            [255, 255, 255]    # white
        ], dtype=np.uint8)

        x = np.arange(width)
        y = np.arange(height)
        idx = (x[np.newaxis, :] + y[:, np.newaxis]) % 8
        img = colors[idx]
        self.original_image = img
        self.current_angle = 0.0
        self.slider.setValue(0)
        self.angle_edit.setText("0")
        self.comp_slider.setValue(0)
        self.comp_angle_edit.setText("0")
        self.selection_rect = None
        self.orig_label.selection_rect = None
        self.split_image_label.selection_rect = None
        self.orig_label.overlay_selection_rect = None
        self.split_image_label.overlay_selection_rect = None
        self.clear_cache()
        self.update_image()
        if self.tabs.currentIndex() == 1:
            self.update_comparison()
        print("Generated Full HD checkerboard pattern")

    # ---------- PSNR Cache Management ----------
    def clear_cache(self):
        self.psnr_cache = {}
        self.cache_params = {}

    def ensure_psnr_cache_params(self):
        return {'a': self.a_value, 'cut': self.zoom_combo_comp.currentText()}

    def is_cache_valid(self):
        params = self.ensure_psnr_cache_params()
        return self.psnr_cache != {} and self.cache_params == params

    def get_methods_list(self):
        base = ['nearest_ref', 'bilinear_ref', 'lanczos_ref']
        if self.show_manual:
            return base + ['nearest_manual', 'bilinear_manual', 'lanczos_manual']
        else:
            return base

    def get_angle_range(self):
        if self.limit_range:
            #h, w = self.original_image.shape[:2] if self.original_image is not None else (0, 0)
            #is_square = (h == w) and h > 0
            max_angle = 45# if is_square else 90
            return list(range(0, max_angle + 1))
        else:
            return list(range(-180, 181))

    def compute_psnr_for_method(self, method, angle, params):
        src = self.original_image
        if src is None:
            return -1.0
        cut = (params['cut'] == 'cut')
        a = params['a']
        if method == 'lanczos_manual':
            fn = lambda img, ang, cut: rotator.rotate_lanczos_manual(img, ang, cut, a)
        else:
            fn = getattr(rotator, f'rotate_{method}', None)
        if fn is None:
            return -1.0
        try:
            rot = fn(src, angle, cut)
            inv = fn(rot, -angle, cut)
            orig_crop, inv_crop, ox0, oy0 = self._align_by_centroid_and_crop(src, inv)
            if orig_crop is None or inv_crop is None or orig_crop.size == 0:
                return -1.0
            if self.selection_rect is not None:
                sel_common = self._map_selection_to_common_with_origin(
                    self.selection_rect, (ox0, oy0), inv_crop.shape[1], inv_crop.shape[0]
                )
                if sel_common is not None:
                    x0, y0, w0, h0 = sel_common
                    if w0 > 0 and h0 > 0:
                        # Make contiguous copies before passing to C++
                        orig_sel = orig_crop[y0:y0+h0, x0:x0+w0].copy()
                        inv_sel = inv_crop[y0:y0+h0, x0:x0+w0].copy()
                        return rotator.psnr(orig_sel, inv_sel)
                return -1.0
            else:
                # No selection: use whole aligned images, but ensure they are contiguous
                return rotator.psnr(orig_crop.copy(), inv_crop.copy())
        except Exception as e:
            print(f"Error computing PSNR for {method} at {angle}: {e}")
            return -1.0

    def compute_method_cache(self, method):
        if method.endswith('_manual') and not self.show_manual:
            return
        params = self.ensure_psnr_cache_params()
        angles = self.get_angle_range()
        self.cache_angles = angles
        self.cache_params = params

        progress = QProgressDialog(f"Computing PSNR for {method}...", "Cancel", 0, len(angles), self)
        progress.setWindowModality(Qt.WindowModal)

        results = []
        for i, angle in enumerate(angles):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            QApplication.processEvents()
            val = self.compute_psnr_for_method(method, angle, params)
            results.append((angle, val))

        progress.setValue(len(angles))
        if not progress.wasCanceled():
            self.psnr_cache[method] = results

    def compute_all_cache(self):
        params = self.ensure_psnr_cache_params()
        methods = self.get_methods_list()
        angles = self.get_angle_range()
        self.cache_angles = angles
        self.cache_params = params

        progress = QProgressDialog("Computing PSNR over angles...", "Cancel", 0, len(angles), self)
        progress.setWindowModality(Qt.WindowModal)

        cache = {m: [] for m in methods}
        for i, angle in enumerate(angles):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            QApplication.processEvents()
            for method in methods:
                val = self.compute_psnr_for_method(method, angle, params)
                cache[method].append((angle, val))

        progress.setValue(len(angles))
        if not progress.wasCanceled():
            self.psnr_cache = cache

    def _filter_psnr_pairs(self, pairs, threshold=1000.0):
        angles = [p[0] for p in pairs]
        values = [p[1] if p[1] <= threshold else np.nan for p in pairs]
        return angles, values

    # ---------- Plotting Methods ----------
    def show_psnr_line_plot(self):
        if not self.is_cache_valid():
            self.compute_all_cache()
        if not self.psnr_cache:
            return
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        methods = self.get_methods_list()
        all_angles = []
        for method in methods:
            if method in self.psnr_cache:
                pairs = self.psnr_cache[method]
                all_angles.extend([p[0] for p in pairs])
        if all_angles:
            min_angle = min(all_angles)
            max_angle = max(all_angles)
            margin = (max_angle - min_angle) * 0.05
            ax.set_xlim(min_angle - margin, max_angle + margin)

        for i, method in enumerate(methods):
            if method in self.psnr_cache:
                pairs = self.psnr_cache[method]
                angles, values = self._filter_psnr_pairs(pairs)
                ax.plot(angles, values, label=method, color=colors[i % len(colors)], marker='.', linestyle='-')

        ax.xaxis.set_major_locator(ticker.MultipleLocator(45))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(15))
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("PSNR after double rotation vs. angle")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        dlg = PlotDialog(fig, "PSNR vs Angle", self)
        dlg.show()

    def show_method_line_plot(self, method):
        params = self.ensure_psnr_cache_params()
        if self.cache_params != params or method not in self.psnr_cache:
            self.compute_method_cache(method)
        if method not in self.psnr_cache:
            return
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        pairs = self.psnr_cache[method]
        angles, values = self._filter_psnr_pairs(pairs)
        ax.plot(angles, values, label=method, marker='.', linestyle='-')

        if angles:
            min_angle = min(angles)
            max_angle = max(angles)
            margin = (max_angle - min_angle) * 0.05
            ax.set_xlim(min_angle - margin, max_angle + margin)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(45))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(15))
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"PSNR after double rotation – {method}")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        dlg = PlotDialog(fig, f"PSNR {method}", self)
        dlg.show()

    def show_psnr_boxplot(self):
        if not self.is_cache_valid():
            self.compute_all_cache()
        if not self.psnr_cache:
            return
        order = self.get_methods_list()
        data = []
        labels = []
        for m in order:
            vals = [v for _, v in self.psnr_cache[m] if 0 <= v <= 1000]
            data.append(vals if vals else [0])
            if m.endswith('_ref'):
                base = m.replace('_ref', '')
                if base == 'lanczos':
                    labels.append('Lanczos ref\n(a=4)')
                else:
                    labels.append(f'{base}\nref')
            else:
                base = m.replace('_manual', '')
                labels.append(f'{base}\nman')

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        colors = []
        for i, m in enumerate(order):
            if m.endswith('_ref'):
                colors.append('lightblue')
            else:
                colors.append('lightcoral')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel("PSNR (dB)")
        ax.set_title("PSNR distribution over angles (infinite values excluded)")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        dlg = PlotDialog(fig, "Boxplot of PSNR", self)
        dlg.show()

    # ---------- UI callbacks ----------
    def on_method_change(self, text):
        self.method = text
        self.update_a_visibility()
        if self.original_image is not None:
            self.update_image()

    def on_zoom_mode_change(self, text):
        self.zoom_mode = text
        # Update comparison combo, block signals to avoid triggering update_comparison
        self.zoom_combo_comp.blockSignals(True)
        self.zoom_combo_comp.setCurrentText(text)
        self.zoom_combo_comp.blockSignals(False)
        self.clear_cache()
        if self.original_image is not None:
            self.update_image()
            if self.tabs.currentIndex() == 1:
                self.update_comparison()

    def on_zoom_change(self, text):
        self.zoom_mode = text
        self.zoom_combo.setCurrentText(text)
        self.clear_cache()
        if self.original_image is not None:
            self.update_comparison()
            self.update_image()

    def on_a_change(self, value):
        self.a_value = value
        self.a_value_label.setText(f"a = {value} (window = {2*value})")
        # Update comparison slider, block signals
        self.comp_a_slider.blockSignals(True)
        self.comp_a_slider.setValue(value)
        self.comp_a_slider.blockSignals(False)
        self.comp_a_label.setText(f"a = {value} (window = {2*value})")
        self.clear_cache()
        if self.original_image is not None:
            if self.method == 'lanczos_manual':
                self.update_image()
            if self.tabs.currentIndex() == 1:
                self.update_comparison()

    def on_comp_a_change(self, value):
        self.a_value = value
        self.comp_a_label.setText(f"a = {value} (window = {2*value})")
        self.clear_cache()
        self.update_comparison()

    def on_slider_change(self, value):
        self.current_angle = float(value)
        self.angle_edit.setText(str(value))
        if self.original_image is not None:
            self.update_image()
        # Update comparison controls without emitting signals
        self.comp_slider.blockSignals(True)
        self.comp_slider.setValue(value)
        self.comp_slider.blockSignals(False)
        self.comp_angle_edit.setText(str(value))
        if self.tabs.currentIndex() == 1:
            self.update_comparison()

    def on_angle_edit(self):
        try:
            angle = float(self.angle_edit.text())
            angle = max(-180.0, min(180.0, angle))
            self.current_angle = angle
            self.slider.setValue(int(round(angle)))
            if self.original_image is not None:
                self.update_image()
            self.comp_slider.blockSignals(True)
            self.comp_slider.setValue(int(round(angle)))
            self.comp_slider.blockSignals(False)
            self.comp_angle_edit.setText(str(angle))
            if self.tabs.currentIndex() == 1:
                self.update_comparison()
        except Exception:
            pass

    def on_comp_angle_change(self, value):
        self.current_angle = float(value)
        self.comp_angle_edit.setText(str(value))
        self.slider.setValue(value)
        self.angle_edit.setText(str(value))
        if self.original_image is not None:
            self.update_comparison()

    def on_comp_angle_edit(self):
        try:
            angle = float(self.comp_angle_edit.text())
            angle = max(-180.0, min(180.0, angle))
            self.current_angle = angle
            self.comp_slider.setValue(int(round(angle)))
            self.slider.setValue(int(round(angle)))
            self.angle_edit.setText(str(angle))
            if self.original_image is not None:
                self.update_comparison()
        except Exception:
            pass

    def on_tab_changed(self, index):
        if index == 1 and self.original_image is not None:
            self.update_comparison()

    def on_split_method_change(self, text):
        self.split_method = text
        if self.original_image is not None:
            self.update_comparison()

    def on_split_slider_change(self, val):
        self.split_image_label.set_split(val)

    # ---------- Rotation pipeline for Rotator tab ----------
    def update_image(self):
        if self.original_image is None:
            return

        start_time = time.perf_counter()
        try:
            cut = (self.zoom_combo.currentText() == 'cut')
            method = self.method_combo.currentText()
            if method == 'lanczos_ref (a=4 fixed)':
                method = 'lanczos_ref'
            if method == 'nearest_ref':
                rotated = rotator.rotate_nearest_ref(self.original_image, self.current_angle, cut)
            elif method == 'bilinear_ref':
                rotated = rotator.rotate_bilinear_ref(self.original_image, self.current_angle, cut)
            elif method == 'lanczos_ref':
                rotated = rotator.rotate_lanczos_ref(self.original_image, self.current_angle, cut)
            elif method == 'nearest_manual':
                rotated = rotator.rotate_nearest_manual(self.original_image, self.current_angle, cut)
            elif method == 'bilinear_manual':
                rotated = rotator.rotate_bilinear_manual(self.original_image, self.current_angle, cut)
            elif method == 'lanczos_manual':
                rotated = rotator.rotate_lanczos_manual(self.original_image, self.current_angle, cut, self.a_value)
            else:
                return

            print("Rotated image:", None if rotated is None else (rotated.shape, rotated.dtype))

            if self.zoom_combo.currentText() == 'zoom_to_content' and rotated is not None and rotated.size != 0:
                iw, ih = rotator.get_max_inner_rect(self.original_image.shape[1], self.original_image.shape[0], self.current_angle)
                iw = max(1, int(iw))
                ih = max(1, int(ih))
                hh, ww = rotated.shape[:2]
                start_y = max(0, (hh - ih) // 2)
                start_x = max(0, (ww - iw) // 2)
                if start_y + ih <= hh and start_x + iw <= ww:
                    rotated = rotated[start_y:start_y+ih, start_x:start_x+iw]

        except Exception as e:
            print(f"Rotation error: {e}")
            return

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        self.time_label.setText(f"Processing time: {elapsed_ms:.2f} ms")

        self.current_image = rotated
        self.show_image_on_label(self.image_label, rotated, max_size=700)

    def resizeEvent(self, event):
        if self.current_image is not None:
            self.show_image_on_label(self.image_label, self.current_image, max_size=self.image_label.width())
        super().resizeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = RotationApp()
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()