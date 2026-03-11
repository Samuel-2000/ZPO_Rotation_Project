# gui.py
import sys
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSlider,
                             QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QFileDialog, QComboBox, QGroupBox,
                             QGridLayout)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator, QPainter, QPen
from PIL import Image
import cpp_rotator.rotator_cpp as rotator


class SelectableLabel(QLabel):
    """QLabel allowing rectangular selection and drawing overlay selection_rect in pixmap coords."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_rect = None                # QRect in pixmap coordinates (user-drawn)
        self.overlay_selection_rect = None        # QRect in pixmap coordinates (externally set)
        self.rubber_band = None
        self.origin = QPoint()
        self.setMouseTracking(True)
        self.selection_callback = None

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
    Builds a combined pixmap and sets it on the label. Keeps numpy refs so QImage buffer remains valid.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_left = None
        self.img_right = None
        self.split_pos = 50  # percentage
        self._left_ref = None
        self._right_ref = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 120)

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
        # debug
        try:
            print("SplitImageLabel.set_images left:", None if self.img_left is None else self.img_left.shape,
                  "right:", None if self.img_right is None else self.img_right.shape)
        except Exception:
            pass
        self._build_and_set_pixmap()

    def set_split(self, pos):
        self.split_pos = max(0, min(100, int(pos)))
        self._build_and_set_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._build_and_set_pixmap()

    def _build_and_set_pixmap(self):
        if self.img_left is None or self.img_right is None:
            self.clear()
            return
        try:
            left = self.img_left
            right = self.img_right

            # choose target area based on label size but preserve aspect via scaling later
            tgt_w = max(1, self.width())
            tgt_h = max(1, self.height())

            qimg_left = QImage(left.data, left.shape[1], left.shape[0], 3 * left.shape[1], QImage.Format_RGB888)
            qimg_right = QImage(right.data, right.shape[1], right.shape[0], 3 * right.shape[1], QImage.Format_RGB888)

            pix_left = QPixmap.fromImage(qimg_left).scaled(tgt_w, tgt_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pix_right = QPixmap.fromImage(qimg_right).scaled(tgt_w, tgt_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

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

            final = canvas.scaled(tgt_w, tgt_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(final)
            # keep refs alive
            self._last_pixmap = final
            self._left_ref = left
            self._right_ref = right
        except Exception as e:
            print("SplitImageLabel: error building pixmap:", e)
            self.clear()


class RotateApp(QMainWindow):
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

        self.initUI()

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

        left.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            'nearest_ref', 'bilinear_ref', 'lanczos_ref',
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
        self.image_label.setMinimumSize(500, 420)
        self.image_label.setStyleSheet("border: 1px solid black")
        layout.addLayout(left, 1)
        layout.addWidget(self.image_label, 4)

        self.update_a_visibility()

    def update_a_visibility(self):
        visible = (self.method == 'lanczos_manual')
        self.a_label.setVisible(visible)
        self.a_slider.setVisible(visible)
        self.a_value_label.setVisible(visible)

    # ---------- Comparison tab ----------
    def init_comparison_tab(self, parent):
        main_layout = QHBoxLayout(parent)
        left_panel = QVBoxLayout()

        self.orig_label = SelectableLabel()
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setMinimumSize(380, 300)
        self.orig_label.setStyleSheet("border: 1px solid black")
        left_panel.addWidget(self.orig_label)
        self.orig_label.selection_callback = self.on_user_selection_changed

        # Zoom preview (unrotated) under original
        self.zoom_preview_label = QLabel()
        self.zoom_preview_label.setAlignment(Qt.AlignCenter)
        self.zoom_preview_label.setMinimumSize(260, 180)
        self.zoom_preview_label.setStyleSheet("border: 1px solid #888;")
        left_panel.addWidget(self.zoom_preview_label)

        # Split preview controls
        split_group = QGroupBox("Zoomed split comparison")
        split_layout = QVBoxLayout(split_group)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.split_method_combo = QComboBox()
        self.split_method_combo.addItems([
            'nearest_ref', 'bilinear_ref', 'lanczos_ref',
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
        for row, (ref_name, man_name) in enumerate(pairs):
            ref_cell = QWidget()
            ref_layout = QVBoxLayout(ref_cell)
            lbl_ref = QLabel(ref_name)
            lbl_ref.setAlignment(Qt.AlignCenter)
            ref_layout.addWidget(lbl_ref)
            ref_img = QLabel()
            ref_img.setAlignment(Qt.AlignCenter)
            ref_img.setMinimumSize(260, 240)
            ref_img.setStyleSheet("border: 1px solid gray")
            ref_layout.addWidget(ref_img)
            ref_psnr = QLabel("PSNR: -- dB")
            ref_psnr.setAlignment(Qt.AlignCenter)
            ref_layout.addWidget(ref_psnr)
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
            man_psnr = QLabel("PSNR: -- dB")
            man_psnr.setAlignment(Qt.AlignCenter)
            man_layout.addWidget(man_psnr)
            grid_layout.addWidget(man_cell, row, 1)

            self.method_cells.append((ref_name, ref_img, ref_psnr, man_name, man_img, man_psnr))

        right_panel.addWidget(grid_group)
        right_panel.addStretch()

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)

    # ---------- Selection mapping ----------
    def on_user_selection_changed(self):
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

        # update overlay and comparison
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

        dx = int(round(cent_o[0] - cent_i[0]))
        dy = int(round(cent_o[1] - cent_i[1]))

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

        # Show original with overlay
        self.show_image_on_label(self.orig_label, self.original_image, max_size=380,
                                 selection=self.selection_rect, set_overlay=True)

        # Set original region for split view (zoom left image)
        if self.selection_rect is not None:
            x, y, w, h = self.selection_rect
            self.zoom_original_region = self.original_image[y:y+h, x:x+w].copy()
        else:
            oh, ow = self.original_image.shape[:2]
            s = max(32, int(min(oh, ow) / 3))
            self.zoom_original_region = self._center_crop(self.original_image, s, s)

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
                            right_img = self._center_crop(inv_crop, zh, zw)
                    else:
                        zh, zw = self.zoom_original_region.shape[:2]
                        right_img = self._center_crop(inv_crop, zh, zw)
        except Exception as e:
            print(f"Split preview error for method {split_method}: {e}")
            right_img = None

        # Update split view widget
        try:
            if right_img is not None and self.zoom_original_region is not None:
                self.split_image_label.set_images(self.zoom_original_region, right_img)
                self.split_image_label.set_split(self.split_slider.value())
        except Exception as e:
            print("Error updating split preview:", e)

        # ---------------------------
        # 3x2 GRID
        # ---------------------------
        for (ref_name, ref_img_label, ref_psnr_label, man_name, man_img_label, man_psnr_label) in self.method_cells:
            try:
                ref_fn = method_funcs[ref_name]
                man_fn = method_funcs[man_name]

                ref_rot = ref_fn(src, angle, cut)
                ref_inv = ref_fn(ref_rot, -angle, cut)

                man_rot = man_fn(src, angle, cut)
                man_inv = man_fn(man_rot, -angle, cut)

                display_ref = ref_inv
                display_man = man_inv

                if use_zoom:
                    iw, ih = rotator.get_max_inner_rect(src.shape[1], src.shape[0], angle)
                    iw = max(1, int(iw))
                    ih = max(1, int(ih))
                    display_ref = self._center_crop(ref_inv, ih, iw)
                    display_man = self._center_crop(man_inv, ih, iw)

                orig_crop_ref, inv_crop_ref, orig_x0_ref, orig_y0_ref = self._align_by_centroid_and_crop(src, display_ref)
                orig_crop_man, inv_crop_man, orig_x0_man, orig_y0_man = self._align_by_centroid_and_crop(src, display_man)

                sel = self.selection_rect

                sel_ref_in_common = self._map_selection_to_common_with_origin(sel, (orig_x0_ref, orig_y0_ref),
                                                                             inv_crop_ref.shape[1], inv_crop_ref.shape[0]) if (orig_crop_ref is not None and inv_crop_ref is not None) else None
                sel_man_in_common = self._map_selection_to_common_with_origin(sel, (orig_x0_man, orig_y0_man),
                                                                             inv_crop_man.shape[1], inv_crop_man.shape[0]) if (orig_crop_man is not None and inv_crop_man is not None) else None

                def psnr_on_region(orig_c, inv_c, sel_common):
                    if orig_c is None or inv_c is None:
                        return -1.0
                    if sel_common is None:
                        return rotator.psnr(orig_c, inv_c)
                    x0, y0, w0, h0 = sel_common
                    if w0 <= 0 or h0 <= 0:
                        return -1.0
                    o_crop = orig_c[y0:y0+h0, x0:x0+w0]
                    i_crop = inv_c[y0:y0+h0, x0:x0+w0]
                    return rotator.psnr(o_crop, i_crop)

                psnr_ref = psnr_on_region(orig_crop_ref, inv_crop_ref, sel_ref_in_common)
                psnr_man = psnr_on_region(orig_crop_man, inv_crop_man, sel_man_in_common)

                ref_psnr_label.setText(f"PSNR: {psnr_ref:.2f} dB" if psnr_ref >= 0 else "PSNR: -- dB")
                man_psnr_label.setText(f"PSNR: {psnr_man:.2f} dB" if psnr_man >= 0 else "PSNR: -- dB")

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
                display_for_label(orig_crop_man, inv_crop_man, (orig_x0_man, orig_y0_man), sel_man_in_common, man_img_label)

            except Exception as e:
                print(f"Error computing {ref_name}/{man_name}: {e}")
                ref_psnr_label.setText("PSNR: error")
                man_psnr_label.setText("PSNR: error")

    # ---------- Show image helper ----------
    def show_image_on_label(self, label, img_np, max_size=120, selection=None, set_overlay=False):
        """Robustly display numpy RGB image on QLabel; keeps reference to array on label to avoid GC."""
        if img_np is None or img_np.size == 0:
            label.clear()
            return
        img_np = np.ascontiguousarray(img_np)
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        # Force 3 channels
        if img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=2)
        elif img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        elif img_np.ndim != 3 or img_np.shape[2] != 3:
            label.clear()
            return

        h, w = img_np.shape[:2]
        bytes_per_line = 3 * w

        # create QImage from buffer and hold reference
        qimg = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label._img_ref = img_np

        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

        # overlay selection if requested and label is SelectableLabel
        if isinstance(label, SelectableLabel):
            if set_overlay and (selection is not None) and (self.original_image is not None):
                orig_h, orig_w = self.original_image.shape[:2]
                pix_w = scaled.width()
                pix_h = scaled.height()
                scale_x = pix_w / orig_w
                scale_y = pix_h / orig_h
                orig_x, orig_y, orig_wd, orig_hd = selection
                sel_x = int(round(orig_x * scale_x))
                sel_y = int(round(orig_y * scale_y))
                sel_w = max(1, int(round(orig_wd * scale_x)))
                sel_h = max(1, int(round(orig_hd * scale_y)))
                label.overlay_selection_rect = QRect(sel_x, sel_y, sel_w, sel_h)
            else:
                label.overlay_selection_rect = None
            label.update()

    # ---------- Load / Save ----------
    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if not fname:
            return
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
        self.orig_label.overlay_selection_rect = None
        self.update_image()
        if self.tabs.currentIndex() == 1:
            self.update_comparison()

    def save_image(self):
        if self.current_image is None:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg)")
        if fname:
            Image.fromarray(self.current_image).save(fname)

    # ---------- UI callbacks ----------
    def on_method_change(self, text):
        self.method = text
        self.update_a_visibility()
        if self.original_image is not None:
            self.update_image()

    def on_zoom_mode_change(self, text):
        self.zoom_mode = text
        self.zoom_combo_comp.setCurrentText(text)
        if self.original_image is not None:
            self.update_image()
            if self.tabs.currentIndex() == 1:
                self.update_comparison()

    def on_zoom_change(self, text):
        self.zoom_mode = text
        self.zoom_combo.setCurrentText(text)
        if self.original_image is not None:
            self.update_comparison()
            self.update_image()

    def on_a_change(self, value):
        self.a_value = value
        self.a_value_label.setText(f"a = {value} (window = {2*value})")
        if self.original_image is not None:
            if self.method == 'lanczos_manual':
                self.update_image()
            if self.tabs.currentIndex() == 1:
                self.update_comparison()

    def on_comp_a_change(self, value):
        self.a_value = value
        self.comp_a_label.setText(f"a = {value} (window = {2*value})")
        self.update_comparison()

    def on_slider_change(self, value):
        self.current_angle = float(value)
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
            angle = max(-180.0, min(180.0, angle))
            self.current_angle = angle
            self.slider.setValue(int(round(angle)))
            if self.original_image is not None:
                self.update_image()
            if self.tabs.currentIndex() == 1:
                self.comp_slider.setValue(int(round(angle)))
                self.comp_angle_edit.setText(str(angle))
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
            method = self.method
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
    w = RotateApp()
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()