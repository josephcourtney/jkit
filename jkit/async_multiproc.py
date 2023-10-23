import asyncio
import signal
import time
from ctypes import c_bool
from multiprocessing import Pipe, Process, Value
from typing import TYPE_CHECKING, Callable, TypeAlias

from multiprocessing.sharedctypes import Synchronized
ValueType: TypeAlias = Synchronized

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

    def __init__(
        self, duration: float, cleanup_func=None, hard_exit=True, clock_type: TimerEnum | str = TimerEnum.REAL
    ) -> None:
        """

        Parameters
        ----------
        duration : float
            duration of time limit in seconds
        clock_type : TimerEnum|str, either TimerEnum.{REAL, USER, TOTAL} or {'REAL', 'USER', 'TOTAL'}
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
            return
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


class Worker:
    """Worker
    A worker process wrapper with both synchronous and asynchronous access.
    """

    def __init__(self, control_flag: ValueType, func: Callable, *args, **kwargs) -> None:
        """
        NOTE: this class should only ever be instantiated through the WorkerManager.run method.

        Parameters
        ----------
        control_flag : multiprocess.Value
        func : Callable
        *args : arguments for func
        **kwargs : keyword arguments for func
        """
        self.control_flag = control_flag
        self.func = func
        self._args = args
        self._kwargs = kwargs

        self.conn_read, self.conn_write = Pipe(duplex=False)
        self.proc = Process(target=self._work)
        self.proc.daemon = True
        self.proc.start()

    def _work(self):
        signal.signal(signal.SIGINT, ignore)
        signal.signal(signal.SIGTERM, ignore)
        func(
            *self._args,
            running_status=self.control_flag,
            connection=self.conn_write,
            **self._kwargs,
        )
        self.conn_write.send(None)

    def _close(self):
        self.conn_read.close()
        self.proc.kill()

    async def _aclose(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._close)

    def _get(self):
        if self.conn_read.poll():
            if msg := self.conn_read.recv():
                return msg
            raise EOFError
        return None

    def read(self, size=-1):
        """
        synchronous method to retrieve the results from the worker process.

        Parameters
        ----------
        size : int
            The amount of data to read. If size == -1, all available data is read.
        """
        buffer = []
        while size < 0 or len(buffer) < size:
            try:
                if msg := self._get():
                    buffer.append(msg)
            except EOFError:
                self._close()
                break
        return buffer

    async def aread(self, size=-1):
        """
        asynchronous method to retrieve the results from the worker process.

        Parameters
        ----------
        size : int
            The amount of data to read. If size == -1, all available data is read.
        """
        buffer = []
        while size < 0 or len(buffer) < size:
            try:
                if msg := self._get():
                    buffer.append(msg)
            except EOFError:
                await self._aclose()
                break
            await asyncio.sleep(0)
        return buffer

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                if msg := self._get():
                    return msg
            except EOFError:
                raise StopIteration from None

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            try:
                if msg := self._get():
                    return msg
            except EOFError:
                raise StopAsyncIteration from None
            await asyncio.sleep(0)


class WorkerManager:
    """WorkerManager."""

    def __init__(self) -> None:
        self.workers = []

        self._running_flags = {}

        self._prev_signal_handlers = {}
        self._prev_signal_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self.handle_signal)
        self._prev_signal_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self.handle_signal)

    def running(self, worker):
        return self._running_flags[worker].value

    def kill(self, worker):
        self._running_flags[worker].value = False

    async def akill(self, worker):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.kill, worker)

    def kill_all(self):
        for worker in self.workers:
            self._running_flags[worker].value = False

    def handle_signal(self, signal, frame):
        self.kill()
        if prev_handler := self._prev_signal_handlers[signal]:
            prev_handler(signal, frame)

    def run(self, func, *args, **kwargs):
        """
        spawn a worker process
        Parameters
        func : Callable
            Function that performs work and returns results. The function must take the following keyword arguments:
                - `running_status: multiprocessing.Value(c_bool)` - `running_status.value == True` indicates that
                  the function should continue. A value of `False` indicates that the process will soon close and the
                  function should write any remaining results and return.
                - `connection: multiprocessing.Connection` - the write-only end of a simplex pipe for communicating
                  results back to the manager.
        *args : arguments supplied to func
        **kwargs : keyword arguments supplied to func
        __________.
        """
        new_flag = Value(c_bool, True)
        new_worker = Worker(new_flag, func, *args, **kwargs)
        self.workers.append(new_worker)
        self._running_flags[new_worker] = new_flag
        return new_worker

    async def arun(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run, func, *args, **kwargs)


def func(max_chunk_size, *, running_status, connection):
    buffer = []
    data_generator = iter(range(10000))
    while running_status.value:
        for _ in range(max_chunk_size):
            datum = next(data_generator)
            buffer.append(datum)
            time.sleep(0.01)
        if buffer:
            connection.send(buffer)
        buffer.clear()


def ignore(*args, **kwargs):  # noqa: ARG001
    pass


async def main():
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, ignore)
    loop.add_signal_handler(signal.SIGTERM, ignore)

    mgr = WorkerManager()

    # synchronous worker launch, read, and kill
    worker_0 = mgr.run(func, 3)
    worker_1 = mgr.run(func, 3)
    worker_2 = await mgr.arun(func, 3)
    worker_3 = await mgr.arun(func, 3)

    print("read result:", worker_0.read(10))

    start = time.perf_counter()
    with TimeLimit(0.3, cleanup_func=lambda: mgr.kill(worker_1)):
        for e in worker_1:
            print(e)
    print(time.perf_counter() - start)


    print("read result:", await worker_2.aread(10))

    start = time.perf_counter()
    with TimeLimit(0.3, cleanup_func=lambda: mgr.kill(worker_3)):
        async for e in worker_3:
            print(e)
    print(time.perf_counter() - start)

    await mgr.akill(worker_0)
    await mgr.akill(worker_2)


if __name__ == "__main__":
    with asyncio.Runner() as runner:
        runner.run(main())
