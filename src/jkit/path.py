import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

_GLOB_CHARS = re.compile(r"[*?[]")


def is_glob(s: str) -> bool:
    return _GLOB_CHARS.search(s) is not None


type PathSpec = str | Path | Iterable[str | Path]


def _flatten(elements: Iterable) -> Iterator[str | Path]:
    for elem in elements:
        if isinstance(elem, str | Path):
            yield elem
        elif isinstance(elem, Iterable) and not isinstance(elem, bytes | str | Path):
            yield from _flatten(elem)
        else:
            msg = f"Unsupported type in path specification: {type(elem)}"
            raise TypeError(msg)


def _process_glob_item(item: str, base_dir: Path) -> Iterator[Path]:
    try:
        for p in base_dir.glob(item):
            try:
                if p.is_file() and os.access(p, os.R_OK):  # Check if file is readable
                    yield p.resolve()
            except PermissionError:
                continue  # Skip inaccessible files
    except PermissionError:
        return  # Skip inaccessible directories


def _process_non_glob_item(item: str | Path) -> Iterator[Path]:
    path = item if isinstance(item, Path) else Path(item)
    try:
        if not path.exists():
            return

        if path.is_file() and os.access(path, os.R_OK):  # Check if file is readable
            yield path.resolve()
        elif path.is_dir():
            try:
                yield from (p.resolve() for p in path.rglob("*") if p.is_file() and os.access(p, os.R_OK))
            except PermissionError:
                return  # Skip inaccessible directories
    except PermissionError:
        return  # Skip inaccessible files or directories


def normalize_paths(input_arg: PathSpec, base_dir: Path | None = None) -> Iterator[Path]:
    if isinstance(input_arg, str | Path):
        items = [input_arg]
    elif isinstance(input_arg, Iterable):
        items = _flatten(input_arg)
    else:
        msg = f"Unsupported type: {type(input_arg)}"
        raise TypeError(msg)

    base_dir = base_dir or Path.cwd()

    for item in items:
        if isinstance(item, str) and is_glob(item):
            yield from _process_glob_item(item, base_dir)
        else:
            yield from _process_non_glob_item(
                item,
            )


def unpropagated_changes(target: PathSpec, source: PathSpec) -> bool:
    """Check if any source file is newer than all target files."""
    source_files = list(normalize_paths(source))
    target_files = list(normalize_paths(target))

    if not source_files or not target_files:
        return False

    try:
        youngest_source = max(f.stat().st_mtime for f in source_files)
        oldest_target = min(f.stat().st_mtime for f in target_files)
    except (ValueError, FileNotFoundError):
        return False

    return youngest_source > oldest_target
