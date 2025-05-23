import itertools
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

import tzlocal

NotDefined = object()


def nop(*args: Any, **kwargs: dict) -> None:
    pass


def chunked_iterator(ary, chunk_shape, overlap):
    slices = [[] for _ in range(len(chunk_shape))]
    for i in range(len(chunk_shape)):
        for j in range(0, ary.shape[i], chunk_shape[i] - overlap[i]):
            slices[i].append(slice(j, min(ary.shape[i] - 1, j + chunk_shape[i])))
    for chnk in itertools.product(*slices):
        yield tuple(slc.start for slc in chnk), ary[chnk]


def format_timedelta(td):
    days = td.days
    seconds = td.seconds + 1e-6 * td.microseconds
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if days > 0:
        parts.append(f"{days:02.0f}")
    if hours > 0:
        parts.append(f"{hours:02.0f}")
    if minutes > 0:
        parts.append(f"{minutes:02.0f}")
    parts.append(f"{seconds:06.3f}")
    return ":".join(parts)


def now_here():
    return datetime.now(tz=tzlocal.get_localzone())


def suppressing_map(
    function: Callable,
    iterable: Iterable,
    exceptions: BaseException | tuple[BaseException, ...] | None = None,
    default: Any = NotDefined,  # noqa: ANN401 # This is intentionally allowed to by anything
) -> Iterable[Any]:
    """
    Apply a function to each element in the iterable, suppressing specified exceptions.

    Parameters
    ----------
    function : Callable
        The function to apply to each element in the iterable.
    iterable : Iterable
        The iterable containing the elements to process.
    exceptions : Exception | tuple[Exception], optional
        The exception types to suppress. If None (default), all exceptions are suppressed.
    default : Any, optional
        The default value to yield when an exception occurs. If not supplied (default), the item is skipped.

    Yields
    ------
    any
        The result of applying the function to each element in the iterable, or the default value if an
            exception occurs.

    Notes
    -----
    If exceptions is not provided or is None, all exceptions are suppressed.
    If default is provided and an exception occurs, the default value is yielded instead of raising the
        exception.
    If default is None, the item is skipped when an exception occurs.

    Examples
    --------
    >>> from math import sqrt
    >>> iterable = [1, 4, 9, -1, 16]
    >>> list(suppressing_map(sqrt, iterable, exceptions=ValueError, default=None))
    [1.0, 2.0, 3.0, None, 4.0]
    """
    if exceptions is None:
        exceptions = (BaseException,)
    for item in iterable:
        try:
            yield function(item)
        except exceptions:
            if default is not NotDefined:
                yield default
            else:
                continue
