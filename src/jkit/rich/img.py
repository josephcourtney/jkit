import warnings
from wskr.rich.img import RichImage  # whatever lives there

warnings.warn(
    "jkit.rich.img is deprecated and has moved to wskr.rich.img; please import from wskr.rich.img instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["RichImage"]
