from __future__ import annotations

from typing import Callable

import AppKit
import Quartz

from contextlib import contextmanager


ease_in = Quartz.CAMediaTimingFunction.functionWithName_(Quartz.kCAMediaTimingFunctionEaseIn)
ease_in_out = Quartz.CAMediaTimingFunction.functionWithName_(Quartz.kCAMediaTimingFunctionEaseInEaseOut)


@contextmanager
def animation_context():
    try:
        AppKit.NSAnimationContext.beginGrouping()
        yield AppKit.NSAnimationContext.currentContext()
    finally:
        AppKit.NSAnimationContext.endGrouping()

class Overlay(AppKit.NSWindow):
    target_alpha: float = 0.9
    fade_in_duration: float = 1.5
    fade_out_duration: float = 0.5
    color: AppKit.NSColor = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 0, 0, 1)

    @staticmethod
    def canBecomeMainWindow() -> bool:
        return True

    @staticmethod
    def canBecomeKeyWindow() -> bool:
        return True

    @classmethod
    def for_screen(cls, screen: AppKit.NSScreen) -> Overlay:
        window = cls.alloc().initWithContentRect_styleMask_backing_defer_(
            screen.frame(),
            AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        window.setLevel_(Quartz.CGWindowLevelForKey(Quartz.kCGOverlayWindowLevelKey))
        window.setAlphaValue_(0.0)
        # window.setBackgroundColor_(cls.color)
        window.setOpaque_(False)
        window.setHasShadow_(False)
        window.setCollectionBehavior_(
            window.collectionBehavior()
            | AppKit.NSWindowCollectionBehaviorStationary
            | AppKit.NSWindowCollectionBehaviorIgnoresCycle
            | AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces,
        )
        return window

    def display(self, finished_callback: Callable | None = None):
        self.orderFrontRegardless()

        with animation_context() as c:
            if finished_callback:
                c.setCompletionHandler_(finished_callback)
            c.setDuration_(self.fade_in_duration)
            c.setTimingFunction_(ease_in_out)
            self.animator().setAlphaValue_(self.target_alpha)

    def hide(self, finished_callback: Callable | None = None):
        def _finished_callback():
            self.orderOut_(self)
            if finished_callback:
                finished_callback()

        self.orderFrontRegardless()

        with animation_context() as c:
            c.setCompletionHandler_(_finished_callback)
            c.setDuration_(self.fade_out_duration)
            c.setTimingFunction_(ease_in)
            self.animator().setAlphaValue_(0.0)
 

