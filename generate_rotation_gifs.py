#!/usr/bin/env python3
"""
Generate a composite rotation GIF with three modes side by side:
- cut
- preserve
- zoom_to_content

Angles: -180 to 177 (step 3°) → 120 frames.
Frame rate: 24 fps (duration ≈ 41.67 ms).
Full resolution, no downscaling.
Columns: each mode is centered in a fixed-width column; gaps between columns.
GIF optimized.
"""

import os
import sys
import numpy as np
from PIL import Image
import subprocess
import platform

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
# Background color for padding (white)
BG_COLOR = (255, 255, 255)

# Gap between columns (in pixels)
GAP_PX = 20

# Frame rate (frames per second)
FPS = 24

# Angle step in degrees (3 → 120 frames)
ANGLE_STEP = 3

# Do NOT scale the image – keep original size
SCALE_FACTOR = 1.0

# Number of colors in the GIF palette (max 256). Lower = smaller file size.
COLORS = 256

# ----------------------------------------------------------------------
# Helper functions for building the C++ module (copied from run.py)
# ----------------------------------------------------------------------
def build_cpp_extension():
    """Compile the C++ module in cpp_rotator directory."""
    print("Building C++ rotation module...")
    cpp_dir = "cpp_rotator"
    if not os.path.exists(cpp_dir):
        print(f"Error: {cpp_dir} directory not found!")
        return False

    original_dir = os.getcwd()
    os.chdir(cpp_dir)

    try:
        result = subprocess.run([
            sys.executable, "setup.py", "build_ext", "--inplace"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("C++ module built successfully!")
            built = [f for f in os.listdir('.') if f.endswith(('.so', '.pyd', '.dll'))]
            print(f"Built files: {', '.join(built)}")
            return True
        else:
            print("Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"Build error: {e}")
        return False
    finally:
        os.chdir(original_dir)

def check_cpp_extension():
    """Check if the C++ module is available."""
    try:
        import cpp_rotator.rotator_cpp
        print("C++ module is available")
        return True
    except ImportError as e:
        print(f"C++ module not available: {e}")
        return False

def add_opencv_dll_path():
    """Add OpenCV DLL path on Windows."""
    if platform.system() != 'Windows':
        return True
    import os
    opencv_root = None
    if 'OpenCV_DIR' in os.environ:
        opencv_root = os.environ['OpenCV_DIR']
    elif 'OPENCV_DIR' in os.environ:
        opencv_root = os.environ['OPENCV_DIR']
    else:
        candidates = [
            r"C:\opencv\build",
            r"C:\Program Files\opencv\build",
            r"C:\tools\opencv\build",
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "include", "opencv2", "opencv.hpp")):
                opencv_root = cand
                break
    if opencv_root:
        bin_candidates = [
            os.path.join(opencv_root, "x64", "vc15", "bin"),
            os.path.join(opencv_root, "x64", "vc16", "bin"),
            os.path.join(opencv_root, "bin"),
        ]
        for bin_dir in bin_candidates:
            if os.path.exists(bin_dir):
                os.add_dll_directory(bin_dir)
                print(f"✅ Added OpenCV DLL path: {bin_dir}")
                return True
    print("⚠️  Could not find OpenCV bin directory.")
    return False

def ensure_cpp_module():
    """Make sure the C++ module is available, build if needed."""
    if not check_cpp_extension():
        if not build_cpp_extension():
            sys.exit(1)
        add_opencv_dll_path()
        if not check_cpp_extension():
            print("C++ module still not available after build!")
            sys.exit(1)

# ----------------------------------------------------------------------
# Main GIF generation
# ----------------------------------------------------------------------
def load_image(path):
    """Load an image as a numpy array (HxWx3 uint8)."""
    try:
        img = Image.open(path).convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def generate_frames(img, mode, angle_range, rotator, scale_factor):
    """
    Generate frames for one zoom mode, optionally downscaled.
    Returns list of PIL.Image objects and the maximum width/height across frames.
    """
    frames = []
    max_w, max_h = 0, 0

    print(f"Generating frames for mode '{mode}'...")
    total = len(angle_range)

    for i, angle in enumerate(angle_range):
        if i % 50 == 0:
            print(f"  Progress: {i}/{total} angles...")

        # Compute rotated image
        if mode == 'cut':
            rotated = rotator.rotate_lanczos_ref(img, angle, True)
        else:
            rotated = rotator.rotate_lanczos_ref(img, angle, False)
            if mode == 'zoom_to_content':
                iw, ih = rotator.get_max_inner_rect(img.shape[1], img.shape[0], angle)
                iw = max(1, int(iw))
                ih = max(1, int(ih))
                h, w = rotated.shape[:2]
                start_y = max(0, (h - ih) // 2)
                start_x = max(0, (w - iw) // 2)
                rotated = rotated[start_y:start_y+ih, start_x:start_x+iw]

        pil_img = Image.fromarray(rotated)
        if scale_factor != 1.0:
            new_size = (int(pil_img.width * scale_factor), int(pil_img.height * scale_factor))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

        frames.append(pil_img)

        w, h = pil_img.size
        if w > max_w: max_w = w
        if h > max_h: max_h = h

    print(f"  Max frame size: {max_w}x{max_h}")
    return frames, max_w, max_h

def pad_to_size(img, target_width, target_height, bg_color):
    """
    Pad image to target width and height, centering it.
    Returns a new image.
    """
    w, h = img.size
    # Horizontal padding
    if w < target_width:
        new_img = Image.new('RGB', (target_width, h), bg_color)
        x_offset = (target_width - w) // 2
        new_img.paste(img, (x_offset, 0))
        img = new_img
    # Vertical padding
    if h < target_height:
        new_img = Image.new('RGB', (img.width, target_height), bg_color)
        y_offset = (target_height - h) // 2
        new_img.paste(img, (0, y_offset))
        img = new_img
    return img

def combine_frames(cut_frame, preserve_frame, zoom_frame,
                   cut_width, preserve_width, zoom_width,
                   target_height, gap, bg_color):
    """
    Combine three frames horizontally, each padded to its fixed column width and the common height.
    Returns a composite PIL image.
    """
    # Pad each frame to its column width and common height
    cut_pad = pad_to_size(cut_frame, cut_width, target_height, bg_color)
    preserve_pad = pad_to_size(preserve_frame, preserve_width, target_height, bg_color)
    zoom_pad = pad_to_size(zoom_frame, zoom_width, target_height, bg_color)

    total_width = cut_width + gap + preserve_width + gap + zoom_width
    composite = Image.new('RGB', (total_width, target_height), bg_color)
    x = 0
    composite.paste(cut_pad, (x, 0))
    x += cut_width + gap
    composite.paste(preserve_pad, (x, 0))
    x += preserve_width + gap
    composite.paste(zoom_pad, (x, 0))
    return composite

def save_gif(frames, output_path, duration_ms, optimize=True, colors=256):
    """Save list of PIL images as a GIF with palette optimization."""
    if not frames:
        print(f"No frames to save for {output_path}")
        return
    print(f"Saving GIF to {output_path}...")

    if colors < 256:
        frames = [f.quantize(colors=colors, method=Image.Quantize.MEDIANCUT) for f in frames]

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=optimize
    )
    print("Done.")

def main():
    # Ensure C++ module is ready
    ensure_cpp_module()
    import cpp_rotator.rotator_cpp as rotator

    # Load image (lena.png or fallback)
    image_path = "lena.png"
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found. Creating a 512x512 gradient image instead.")
        # Create a simple gradient as fallback
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            for j in range(512):
                img[i,j] = [int(i*255/512), int(j*255/512), 128]
    else:
        img = load_image(image_path)
        if img is None:
            print("Failed to load image, exiting.")
            sys.exit(1)

    # Angles: -180 to 177 with step ANGLE_STEP
    angles = list(range(-180, 180, ANGLE_STEP))
    # Ensure we include the last possible angle
    if angles[-1] < 177:
        angles.append(177)

    # Duration per frame
    frame_duration_ms = int(1000 / FPS)

    # Generate frames for each mode
    modes = ['cut', 'preserve', 'zoom_to_content']
    mode_frames = {}
    mode_widths = []
    mode_heights = []
    for mode in modes:
        frames, max_w, max_h = generate_frames(img, mode, angles, rotator, SCALE_FACTOR)
        mode_frames[mode] = frames
        mode_widths.append(max_w)
        mode_heights.append(max_h)

    # Global maximum height across all modes
    global_max_height = max(mode_heights)
    print(f"Global max height: {global_max_height}")

    # Use fixed column widths: the max width for each mode
    cut_width, preserve_width, zoom_width = mode_widths
    print(f"Column widths: cut={cut_width}, preserve={preserve_width}, zoom={zoom_width}")

    # Build composite frames
    composite_frames = []
    total_angles = len(angles)
    for i in range(total_angles):
        if i % 30 == 0:
            print(f"Building composite frame {i}/{total_angles}")
        composite = combine_frames(
            mode_frames['cut'][i],
            mode_frames['preserve'][i],
            mode_frames['zoom_to_content'][i],
            cut_width, preserve_width, zoom_width,
            global_max_height,
            GAP_PX,
            BG_COLOR
        )
        composite_frames.append(composite)

    # Save composite GIF
    output_filename = "rotation_all_modes.gif"
    save_gif(composite_frames, output_filename, frame_duration_ms,
             optimize=True, colors=COLORS)

if __name__ == "__main__":
    main()