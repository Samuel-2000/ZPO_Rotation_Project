from setuptools import setup, Extension
import pybind11
import os
import sys
import platform
import glob

def find_opencv_windows():
    """Nájde OpenCV na Windows podľa OpenCV_DIR alebo typických ciest."""
    # 1. Skúsime premennú prostredia
    opencv_root = None
    if 'OpenCV_DIR' in os.environ:
        opencv_root = os.environ['OpenCV_DIR']
    elif 'OPENCV_DIR' in os.environ:
        opencv_root = os.environ['OPENCV_DIR']
    else:
        # 2. Typické cesty
        candidates = [
            r"C:\opencv\build",
            r"C:\Program Files\opencv\build",
            r"C:\tools\opencv\build",
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "include", "opencv2", "opencv.hpp")):
                opencv_root = cand
                break

    if opencv_root is None:
        raise RuntimeError(
            "OpenCV nebol nájdený. Nastavte premennú prostredia OpenCV_DIR\n"
            "na koreňový adresár OpenCV (napr. C:\\opencv\\build) a skúste znova.\n"
            "Ak OpenCV nemáte, stiahnite si ho z https://opencv.org/releases/ a nainštalujte."
        )

    # Hľadáme hlavičky
    inc_dir = os.path.join(opencv_root, "include")
    if not os.path.exists(os.path.join(inc_dir, "opencv2", "opencv.hpp")):
        raise RuntimeError(f"Hlavičky OpenCV sa nenašli v {inc_dir}")

    # Hľadáme knižnice – typicky v x64/vc15/lib alebo x64/vc16/lib
    lib_candidates = [
        os.path.join(opencv_root, "x64", "vc15", "lib"),
        os.path.join(opencv_root, "x64", "vc16", "lib"),
        os.path.join(opencv_root, "lib"),
    ]
    lib_dir = None
    for cand in lib_candidates:
        if os.path.exists(cand):
            lib_dir = cand
            break
    if lib_dir is None:
        raise RuntimeError(f"Adresár s knižnicami OpenCV sa nenašiel v {opencv_root}")

    # Nájdeme všetky knižnice opencv_*.lib (napr. opencv_core450.lib)
    lib_files = glob.glob(os.path.join(lib_dir, "opencv_*.lib"))
    if not lib_files:
        raise RuntimeError(f"Žiadne OpenCV knižnice (*.lib) v {lib_dir}")

    # Vrátime zoznam ciest ku knižniciam
    return [f"/I{inc_dir}"], lib_files

def get_opencv_flags():
    """Vráti include a link flagy pre OpenCV podľa platformy."""
    if platform.system() == 'Windows':
        return find_opencv_windows()
    else:
        # Linux / Mac – použijeme pkg-config
        import subprocess
        try:
            cflags = subprocess.check_output(["pkg-config", "--cflags", "opencv4"]).decode().strip().split()
            libs = subprocess.check_output(["pkg-config", "--libs", "opencv4"]).decode().strip().split()
            return cflags, libs
        except:
            # Fallback
            return ['-I/usr/include/opencv4'], ['-lopencv_core', '-lopencv_imgproc', '-lopencv_highgui']

def get_optimization_flags():
    """Optimalizačné flagy podľa platformy."""
    if platform.system() == 'Windows':
        return ['/O2', '/std:c++17', '/fp:fast', '/arch:AVX2']
    else:
        flags = ['-O3', '-march=native', '-ffast-math', '-fopenmp',
                 '-funroll-loops', '-std=c++17']
        if platform.machine() in ['x86_64', 'amd64']:
            flags.extend(['-msse4.2', '-mavx', '-mfma'])
        return flags

# Získanie flagov
opencv_cflags, opencv_libs = get_opencv_flags()
opt_flags = get_optimization_flags()

ext_modules = [
    Extension(
        "rotator_cpp",
        sources=["binding.cpp", "rotation.cpp"],
        include_dirs=[
            ".",
            pybind11.get_include(),
        ] + [flag[2:] for flag in opencv_cflags if flag.startswith('-I')],
        language='c++',
        extra_compile_args=opt_flags + [flag for flag in opencv_cflags if not flag.startswith('-I')],
        extra_link_args=opencv_libs + (['/openmp'] if platform.system() == 'Windows' else ['-fopenmp']),
    )
]

setup(
    name="rotator_cpp",
    version="1.0.0",
    description="C++ image rotation module with pybind11",
    ext_modules=ext_modules,
    zip_safe=False,
)