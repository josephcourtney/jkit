import sys
import types
import warnings

from . import (
    amino_acids,
    plot,
    util,
    warn_with_traceback,
)

__all__ = [
    "amino_acids",
    "plot",
    "util",
    "warn_with_traceback",
]


def create_proxy_for_renamed_subpackage(sub_package, old_sub_package_name):
    # Determine the current package name dynamically
    current_package = sub_package.__name__.rsplit(".", 1)[0]
    full_old_sub_package_name = f"{current_package}.{old_sub_package_name}"

    # Create a proxy module for the old sub-package name
    types.ModuleType(full_old_sub_package_name)

    class ProxyModule(types.ModuleType):
        def __getattr__(self, name):
            msg = " ".join([
                f"The '{full_old_sub_package_name}' module is deprecated and "
                "will be removed in a future release."
                f" Please use '{sub_package.__name__}' instead.",
            ])
            warnings.warn(
                msg,
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(sub_package, name)

    # Insert the proxy module into sys.modules
    sys.modules[full_old_sub_package_name] = ProxyModule(full_old_sub_package_name)
