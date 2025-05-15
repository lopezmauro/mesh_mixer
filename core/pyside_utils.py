import functools
from shiboken2 import wrapInstance
from PySide2 import QtWidgets, QtCore 


def wait_cursor(func):
    """Decorator to show a wait cursor during a long operation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        app = QtWidgets.QApplication.instance()
        if app:
            app.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            return func(*args, **kwargs)
        finally:
            if app:
                app.restoreOverrideCursor()
    return wrapper