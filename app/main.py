"""
Dashcam Speed Analyzer - PySide6 Desktop Application Entrypoint (Phase 12).
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    try:
        from PySide6.QtWidgets import QApplication
        from app.windows.main_window import MainWindow
    except ImportError:
        print("[ERROR] PySide6 is not installed. Run: pip install PySide6", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("Dashcam Speed Analyzer")
    app.setOrganizationName("Antigravity CV Lab")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
