from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

import AppKit

from jkit.macos.blocking_alert import BlockingAlertManager
from jkit.macos.timer import Timer
from jkit.util import now


def nop(*args, **kwargs) -> None:  # noqa: ARG001
    pass


class Alarm:
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        app: AppKit.NSApplication,
        time: datetime | None = None,
        snooze: timedelta | float | None = None,
        title="Alarm",
        message="",
    ) -> None:
        self.app = app
        self.time = time if time is not None else now()
        if snooze is None:
            self.snooze_enabled = False
            self.snooze_duration = 0.0
        else:
            self.snooze_enabled = True
            if isinstance(snooze, timedelta):
                self.snooze_duration = snooze.total_seconds()
            else:
                self.snooze_duration = float(snooze)

        self.title = title
        self.message = message

        self.blocking_alert_manager = BlockingAlertManager()
        self.alert_active = False

        self.timer = None
        self.time = time
        self.start(self.time)

        self.snooze_callback = nop
        self.dismiss_callback = nop

    def register_snooze_callback(self, callback: Callable):
        self.snooze_callback = callback

    def register_dismiss_callback(self, callback: Callable):
        self.dismiss_callback = callback

    def start(self, time):
        self.timer = Timer(start=time, callback=self.alert)

    def snooze(self):
        self.alert_active = False
        self.timer.stop()
        new_alarm_time = now() + timedelta(seconds=self.snooze_duration)
        snooze_delay = new_alarm_time - self.time
        self.snooze_callback(self.snooze_duration, new_alarm_time, snooze_delay)
        self.start(new_alarm_time)

    def dismiss(self):
        self.alert_active = False
        self.timer.stop()
        dismiss_delay = now() - self.time
        self.dismiss_callback(dismiss_delay)
        self.blocking_alert_manager.when_finished(self.app.terminate_)

    def alert(self):
        buttons = {}
        if self.snooze_enabled:
            buttons["Snooze"] = self.snooze
        buttons["Dismiss"] = self.dismiss

        if self.alert_active:
            return

        self.alert_active = True
        self.blocking_alert_manager.display(
            self.title,
            self.message,
            buttons,
            AppKit.NSScreen.mainScreen(),
        )
