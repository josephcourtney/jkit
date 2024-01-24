from __future__ import annotations

from typing import Callable, Iterable

import AppKit

from jkit.macos.overlay import Overlay


def nop(*args, **kwargs) -> None:  # noqa: ARG001
    pass


class BlockingAlertManager:
    overlays: dict[AppKit.NSScreen, Overlay]

    def __init__(self) -> None:
        self.overlays = {}
        for screen in AppKit.NSScreen.screens():
            self.overlays[screen] = Overlay.for_screen(screen)
        self.active_overlays = 0
        self.finished_callback = nop

    def overlay_added(self):
        self.active_overlays += 1

    def overlay_removed(self):
        self.active_overlays -= 1
        if self.active_overlays <= 0:
            self.finished_callback(self)

    def add_overlays(self) -> None:
        for ovl in self.overlays.values():
            ovl.display(self.overlay_added)

    def remove_overlays(self) -> None:
        for ovl in self.overlays.values():
            ovl.hide(self.overlay_removed)

    def when_finished(self, callback: Callable) -> None:
        self.finished_callback = callback

    def display(
        self,
        title: str,
        message: str,
        buttons: Iterable,
        screen: AppKit.NSScreen,
    ) -> None:
        callbacks: dict[int, Callable] = {}
        labels = {}
        if isinstance(buttons, dict):
            for text, callback in buttons.items():
                match text:
                    case "ok" | "OK":
                        response = AppKit.NSModalResponseOK
                    case "cancel" | "Cancel":
                        response = AppKit.NSModalResponseCancel
                    case _:
                        response = [
                            AppKit.NSAlertFirstButtonReturn,
                            AppKit.NSAlertSecondButtonReturn,
                            AppKit.NSAlertThirdButtonReturn,
                        ][len(callbacks)]
                callbacks[response] = callback
                labels[response] = text

        alert = AppKit.NSAlert.alloc().init()
        alert.setAlertStyle_(AppKit.NSWarningAlertStyle)
        alert.setMessageText_(title)
        alert.setInformativeText_(message)
        for text in buttons:
            alert.addButtonWithTitle_(text)

        window = alert.window()
        window.setCollectionBehavior_(
            window.collectionBehavior()
            | AppKit.NSWindowCollectionBehaviorStationary
            | AppKit.NSWindowCollectionBehaviorIgnoresCycle
            | AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces,
        )

        def completionHandler_(returnCode: AppKit.NSModalResponse) -> None:
            alert.window().close()
            self.remove_overlays()
            callbacks.get(returnCode, lambda: returnCode)()

        self.add_overlays()

        alert.beginSheetModalForWindow_completionHandler_(
            self.overlays[screen],
            completionHandler_,
        )
