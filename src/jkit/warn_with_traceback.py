import pathlib
import sys
import textwrap
import traceback
import warnings


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):  # noqa: ARG001
    line_length = 70
    log = file if hasattr(file, "write") else sys.stderr
    st = traceback.extract_stack()
    log.write(f"\n/{'~' * 29}(Warning){'~' * 30}\\\n")
    log.write("\n")
    log.write("\n".join(e.strip() for e in textwrap.wrap(str(message), width=line_length)))
    log.write("\n\n")
    st = traceback.StackSummary.from_list([
        e
        for e in st
        if not (
            pathlib.Path(e.filename).is_relative_to(pathlib.Path(sys.executable).parent.parent)
            or pathlib.Path(e.filename).is_relative_to(pathlib.Path.home() / ".local")
        )
        and e.name != "warn_with_traceback"
    ])
    log.write(
        "\n".join(
            e_1[: line_length - 3] + "..." if len(e_1) > line_length else e_1
            for e_0 in traceback.format_list(st)
            for e_1 in e_0.split("\n")
        )
    )
    log.write(f"\n\\{'~' * 29}(Warning){'~' * 30}/\n\n")


def enable():
    warnings.showwarning = warn_with_traceback
