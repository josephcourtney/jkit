import base64
import fcntl
import io
import math
import sys
import termios
import time
import zlib
from array import array
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from shutil import get_terminal_size

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import subprocess  # noqa: S404 # each use is secure as long as PATH environment variable hasn't been manipulated
import sys
from io import BytesIO

import matplotlib.pyplot as plt

# private name import is OK since this is implementing a backend which requires access to matplotlib internals
from matplotlib import _api, interactive, is_interactive  # noqa: PLC2701
from matplotlib._pylab_helpers import Gcf  # noqa: PLC2701
from matplotlib.backend_bases import FigureManagerBase, _Backend  # noqa: PLC2701
from matplotlib.backends.backend_agg import FigureCanvasAgg

plt.style.use("dark_background")

# heuristic for interactive repl
if sys.flags.interactive:
    interactive(True)  # noqa: FBT003 # defined in matplotlib api


class FigureManagerICat(FigureManagerBase):
    def show(self):
        tput_path = shutil.which("tput")
        tput_lines = subprocess.run(
            [tput_path, "lines"],  # noqa: S603
            capture_output=True,
            text=True,
            check=False,
        )
        rows = int(tput_lines.stdout.rstrip())

        kitty_path = shutil.which("kitty")
        icat_px = subprocess.run(
            [  # noqa: S603
                kitty_path,
                "+kitten",
                "icat",
                "--print-window-size",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        px = [int(e) for e in icat_px.stdout.rstrip().split("x")]
        px[1] -= (3 * px[1]) // rows

        dpi = self.canvas.figure.dpi
        w_raw, h_raw = px[0] / dpi, px[1] / dpi

        w_mult = self.canvas.figure.get_figwidth()
        h_mult = self.canvas.figure.get_figheight()
        aspect_ratio = h_mult / w_mult

        if plt.rcParams["figure.figsize"] == [w_mult, h_mult]:
            if aspect_ratio > 1:
                self.canvas.figure.set_size_inches(h_raw / aspect_ratio, h_raw)
            else:
                self.canvas.figure.set_size_inches(w_raw, w_raw * aspect_ratio)
        else:
            self.canvas.figure.set_size_inches(w_raw * w_mult, h_raw * h_mult)

        with BytesIO() as buf:
            self.canvas.figure.savefig(buf, format="png", transparent=True)
            subprocess.run(
                [  # noqa: S603
                    kitty_path,
                    "+kitten",
                    "icat",
                    "--align",
                    "center",
                ],
                input=buf.getbuffer(),
                stdout=sys.stderr,  # output ot stderr to avoid being piped by default
                check=False,
            )


class FigureCanvasICat(FigureCanvasAgg):
    manager_class = _api.classproperty(lambda _: FigureManagerICat)


@_Backend.export
class _BackendICatAgg(_Backend):
    FigureCanvas = FigureCanvasICat
    FigureManager = FigureManagerICat

    # `draw_if_interactive` isn't really intended for
    # one-shot rendering. We run the risk of being called
    # on a figure that isn't completely rendered yet, so
    # we skip draw calls for figures that we detect as
    # not being fully initialized yet. Our heuristic for
    # that is the presence of axes on the figure.
    @classmethod
    def draw_if_interactive(cls):
        manager = Gcf.get_active()
        if is_interactive() and manager.canvas.figure.get_axes():
            cls.show()

    @classmethod
    def show(cls, *args, **kwargs):
        _Backend.show(*args, **kwargs)
        Gcf.destroy_all()
