#!/usr/bin/env python3
import os
import sys
import subprocess
import platform

def build_cpp_extension():
    """Zkompiluje C++ modul v adresári cpp_rotator."""
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
    """Ověří, zda je C++ modul dostupný (stačí naimportovať modul)."""
    try:
        import cpp_rotator.rotator_cpp
        print("C++ module is available")
        return True
    except ImportError as e:
        print(f"C++ module not available: {e}")
        return False

def install_requirements():
    """Nainštaluje Python závislosti z requirements.txt."""
    if not os.path.exists("requirements.txt"):
        print("requirements.txt not found, skipping.")
        return True
    print("Installing Python requirements...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print("Requirements installed successfully!")
        return True
    else:
        print("Failed to install requirements!")
        print(result.stderr)
        return False

def add_opencv_dll_path():
    """Pridá cestu k OpenCV DLL do vyhľadávania knižníc (Windows)."""
    if platform.system() != 'Windows':
        return True  # Na Linuxe/Mac sa rieši inak
    import os
    opencv_root = None
    # Skúsime premennú prostredia
    if 'OpenCV_DIR' in os.environ:
        opencv_root = os.environ['OpenCV_DIR']
    elif 'OPENCV_DIR' in os.environ:
        opencv_root = os.environ['OPENCV_DIR']
    else:
        # Typické cesty
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
                print(f"✅ Pridaná cesta k OpenCV DLL: {bin_dir}")
                return True
    print("⚠️  Nepodarilo sa nájsť OpenCV bin adresár.")
    print("   Skúste manuálne pridať cestu k OpenCV bin do PATH.")
    return False

def main():
    print("Image Rotator with C++ core")
    print("=" * 50)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")

    # Inštalácia požiadaviek
    if not install_requirements():
        print("Continuing anyway...")

    # Kontrola / kompilácia C++ modulu
    if not check_cpp_extension():
        if not build_cpp_extension():
            sys.exit(1)
        # Po úspešnom builde pridáme cestu k DLL (Windows)
        add_opencv_dll_path()
        if not check_cpp_extension():
            print("C++ module still not available after build!")
            sys.exit(1)

    # Spustenie GUI
    try:
        from PyQt5.QtWidgets import QApplication
        from gui import RotateApp
        app = QApplication(sys.argv)
        gui = RotateApp()
        gui.show()
        sys.exit(app.exec_())
    except ImportError as e:
        print(f"Failed to import GUI: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()