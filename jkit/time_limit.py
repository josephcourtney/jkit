import signal
import time
from enum import Enum, auto
from types import FrameType
from typing import Any, Callable, ClassVar


class TimerEnum(Enum):
    REAL = auto()
    USER = auto()
    TOTAL = auto()


class TimeLimit:
    """TimeLimit.
    A context manager that forcibly stops execution of its contents when a time limit is reached.
    """

    _itimer_choices: ClassVar[dict[TimerEnum, tuple[int, signal.Signals]]] = {
        TimerEnum.REAL: (signal.ITIMER_REAL, signal.SIGALRM),
        TimerEnum.USER: (signal.ITIMER_VIRTUAL, signal.SIGVTALRM),
        TimerEnum.TOTAL: (signal.ITIMER_PROF, signal.SIGPROF),
    }

    class TimesUpError(Exception):
        ...

    def __init__(self, duration: float, cleanup_func=None, hard_exit=True, clock_type: TimerEnum | str = TimerEnum.REAL):
        """

        Parameters
        ----------
        duration : float
            duration of time limit in seconds
        clock_type : TimerEnum|str, either TimerEnum.{REAL, USER, TOTAL} or '{REAL|USER|TOTAL}'
            The type of clock to use for timing execution. REAL measures real time. USER only measures
            time while the process is executing. TOTAL measures time both when the process is executing
            and when the system is executing on behalf of the process.
        """
        self.duration: float = duration
        self.cleanup_func = cleanup_func
        self.clock_type: TimerEnum = TimerEnum(clock_type)
        self.hard_exit = hard_exit

        self._itimer: int
        self._signal: signal.Signals
        self._itimer, self._signal = TimeLimit._itimer_choices[self.clock_type]
        self._prev_handler: Callable[[int, FrameType | None], Any] | int | None

    def _close(self, signal: int, stack: FrameType | None) -> None:
        if self.cleanup_func:
            self.cleanup_func()
        if self.hard_exit:
            raise TimeLimit.TimesUpError

    def __enter__(self) -> None:
        self._prev_handler = signal.signal(
            self._signal,
            self._close,
        )
        signal.setitimer(self._itimer, self.duration)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        signal.signal(self._signal, self._prev_handler)
        return exc_type is None or exc_type == TimeLimit.TimesUpError


if __name__ == '__main__':
    start = time.time()
    with TimeLimit(5.5):
        while True:
            print("Waiting...", time.time() - start)
            time.sleep(1)
    print(time.time() - start)
