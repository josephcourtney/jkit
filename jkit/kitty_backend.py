import os
import sys

from io import BytesIO
import subprocess

from matplotlib import interactive, is_interactive
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (_Backend, FigureManagerBase)
from matplotlib.backends.backend_agg import FigureCanvasAgg

import matplotlib.pyplot as plt
plt.style.use('dark_background')

# XXX heuristic for interactive repl
if sys.flags.interactive:
    interactive(True)


class FigureManagerICat(FigureManagerBase):
    def show(self):
        tput_lines = subprocess.run(['tput', 'lines'], capture_output=True, text=True)
        rows = int(tput_lines.stdout.rstrip())

        icat_px = subprocess.run(['kitty', '+kitten', 'icat', '--print-window-size'], capture_output=True, text=True)
        px = [int(e) for e in icat_px.stdout.rstrip().split('x')]
        px[1] -= (3 * px[1])//rows

        dpi = self.canvas.figure.dpi
        w_raw, h_raw = px[0]/dpi, px[1]/dpi

        aspect_ratio = self.canvas.figure.get_figheight() / self.canvas.figure.get_figwidth()
        if aspect_ratio > 1:
            self.canvas.figure.set_size_inches(h_raw / aspect_ratio, h_raw)
        else:
            self.canvas.figure.set_size_inches(w_raw, w_raw * aspect_ratio)

        with BytesIO() as buf:
            self.canvas.figure.savefig(buf, format='png', transparent=True)
            subprocess.run(['kitty', '+kitten', 'icat', '--align', 'left'], input=buf.getbuffer())


class FigureCanvasICat(FigureCanvasAgg):
    manager_class = FigureManagerICat


@_Backend.export
class _BackendICatAgg(_Backend):

    FigureCanvas = FigureCanvasICat
    FigureManager = FigureManagerICat

    # Noop function instead of None signals that
    # this is an "interactive" backend
    mainloop = lambda: None

    # XXX: `draw_if_interactive` isn't really intended for
    # on-shot rendering. We run the risk of being called
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
