import warnings
from wskr.kitty_backend import FigureCanvasICat, FigureManagerICat, _BackendICatAgg  # adjust names as needed

warnings.warn(
    "jkit.kitty_backend is deprecated and has moved to the wskr package; "
    "please import from wskr.kitty_backend instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["FigureCanvasICat", "FigureManagerICat", "_BackendICatAgg"]
