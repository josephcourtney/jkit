from datetime import timedelta

import numpy as np

from jkit.util import chunked_iterator, format_timedelta, nop, now_here, suppressing_map


def test_nop():
    assert nop() is None


def test_chunked_iterator():
    ary = np.arange(16).reshape(4, 4)
    chunks = list(chunked_iterator(ary, (2, 2), (1, 1)))
    assert len(chunks) > 0


def test_format_timedelta():
    td = timedelta(days=1, hours=2, minutes=3, seconds=4)
    formatted = format_timedelta(td)
    assert formatted == "01:02:03:04.000"


def test_now_here():
    now = now_here()
    assert now.tzinfo is not None


def test_suppressing_map():
    iterable = [1, 2, "a", 4]
    result = list(suppressing_map(int, iterable, exceptions=(ValueError,), default=None))
    assert result == [1, 2, None, 4]


# Add more tests to cover the functions in util.py
