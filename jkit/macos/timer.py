from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

import tzlocal
from Foundation import (
    NSDate,
    NSDefaultRunLoopMode,
    NSRunLoop,
    NSTimer,
)


class Timer:
    repeat_interval: float
    _nsdate: NSDate
    _repeats: bool
    _callback: Callable
    _nstimer: NSTimer

    def __init__(
        self,
        start: datetime | None = None,
        repeat_interval: timedelta | float | None = None,
        callback=None,
    ) -> None:
        if start is None:
            self._nsdate = NSDate.date()
        else:
            self._nsdate = NSDate.dateWithTimeIntervalSince1970_(start.timestamp())

        if isinstance(repeat_interval, timedelta):
            repeat_interval = repeat_interval.total_seconds()

        if repeat_interval is None or repeat_interval <= 0:
            self.repeat_interval = -1
            self._repeats = False
        else:
            self.repeat_interval = repeat_interval
            self._repeats = True

        if callback is None:
            self._callback = lambda: None
        else:
            self._callback = callback

        self._nstimer = NSTimer.alloc().initWithFireDate_interval_target_selector_userInfo_repeats_(
            self._nsdate,
            self.repeat_interval,
            self,
            "callback:",
            None,
            self._repeats,
        )
        NSRunLoop.currentRunLoop().addTimer_forMode_(
            self._nstimer,
            NSDefaultRunLoopMode,
        )

    def callback_(self, _):
        return self._callback()

    def stop(self):
        self._nstimer.invalidate()
        self._nstimer = None
        self._nsdate = None

