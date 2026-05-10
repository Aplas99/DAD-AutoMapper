from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .ui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Dead as Disco Auto Mapper")
    window = MainWindow()
    window.show()
    return app.exec()
