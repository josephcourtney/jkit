# In compliance with the Python Software Foundation License Version 2, the following is reproduced:
#
# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
# 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 Python Software Foundation;
# All Rights Reserved
#
# GenericAlias, _check_methods, and the definition of GeneralIterable are derived from
# cpython/Lib/_collections_abc.py, Python version ~3.12.0rc1, (git short hash c163d7f)
#
# parts of the definition of FriendlyTaskGroup are derived from the definition of TaskGroup in
# cpython/Lib/asyncio/taskgroups.py, Python version ~3.12.0rc1, (git short hash c163d7f)
#
# the definition of merge is derived from the roundrobin() recipe in the itertools documentation
# cpython/Doc/library/itertools.rst, Python version ~3.12.0rc1, (git short hash c163d7f)


import asyncio
from abc import ABCMeta
from asyncio import Task, TaskGroup
from itertools import cycle, islice
from collections.abc import Iterable
from contextvars import Context
from typing import TypeVar, TypeAlias, Coroutine, Any, Generator
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsNext, SupportsAnext



GenericAlias = type(list[int])
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)

# Aliases imported by multiple submodules in typeshed
if sys.version_info >= (3, 12):
    _CoroutineLike: TypeAlias = Coroutine[Any, Any, _T]
else:
    _CoroutineLike: TypeAlias = Generator[Any, None, _T] | Coroutine[Any, Any, _T]

# taken from cpython/Lib/_collections_abc.py
def _check_methods(cls, method):
    mro = cls.__mro__
    for base_class in mro:
        if method in base_class.__dict__:
            if base_class.__dict__[method] is None:
                return NotImplemented
            return True
    return NotImplemented


class GeneralIterable(metaclass=ABCMeta):
    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is GeneralIterable:
            return _check_methods(subclass, "__iter__", "__aiter__")
        return NotImplemented

    __class_getitem__: classmethod = classmethod(GenericAlias)


class FriendlyTaskGroup(TaskGroup):
    """FriendlyTaskGroup."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def loop(self):
        return self._loop

    def _check_accepting_tasks(self, raise_exception=True):
        if not self._entered:
            if raise_exception:
                msg = f"TaskGroup {self!r} has not been entered"
                raise RuntimeError(msg)
            return False

        if self._exiting and not self._tasks:
            if raise_exception:
                msg = f"TaskGroup {self!r} is finished"
                raise RuntimeError(msg)
            return False
        if self._aborting:
            if raise_exception:
                msg = f"TaskGroup {self!r} is shutting down"
                raise RuntimeError(msg)
            return False

        return True

    def add_task(self, task: Task) -> Task:
        """Add a task in this task group.

        Parameters
        ----------
        task : asyncio.Task
            The task to add to this group.

        Returns
        ----------
        task : asyncio.Task
            The same task provided as input.
        """
        self._check_accepting_tasks()  # strangely, mypy does not complain about this even though "_check_accepting_tasks" is not defined in the typeshed stub
        # optimization: Immediately call the done callback if the task is
        # already done (e.g. if the coro was able to complete eagerly),
        # and skip scheduling a done callback
        if task.done():
            # mypy error: "FriendlyTaskGroup" has no attribute "_on_task_done"  [attr-defined]
            self._on_task_done(task)
        else:
            # mypy error: "FriendlyTaskGroup" has no attribute "_tasks"  [attr-defined]
            self._tasks.add(task)
            # mypy error: "FriendlyTaskGroup" has no attribute "_on_task_done"  [attr-defined]
            task.add_done_callback(self._on_task_done)
        return task

    def create_task(
        self,
        coro: _CoroutineLike[_T],
        *,
        name: str | None = None,
        context: Context | None = None,
    ) -> Task[_T]:
        """Create a task in this task group. The signature matches that of `asyncio.create_task().`

        Parameters
        ----------
        coro : Coroutine
            The coroutine to wrap in a `Task` and schedule for execution.
        name : str, optional
            The name given to the task using `Task.set_name()`.
        context : contextvars.Context, optional
            The Context for coro to run in. If no context is provided, a copy of the current context is used.
        """

        task = (
            self._loop.create_task(coro) if context is None else self._loop.create_task(coro, context=context)
        )  # mypy error: "FriendlyTaskGroup" has no attribute "_loop"  [attr-defined]
        task.set_name(name)
        return self.add_task(task)



def merge(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

async def make_iterable_async(it: SupportsNext) -> SupportsAnext:
    """make an async iterable from a synchronous one by awaiting before each item

    Parameters
    ----------
    it : Iterable
        synchronous iterable

    Yields
    ----------
    _ : Any
        elements yielded from the input iterable

    """
    for item in it:
        await asyncio.sleep(0)
        yield item



async def amerge(
    *iterables: Iterable[SupportsAnext | SupportsNext],
):
    """merge async iterables into a one

    Parameters
    ----------
    async_iterables :
        async iterable objects

    Yields
    ----------
    _ : Any
        elements yielded from each of the input iterables in the order they are chosen by the event loop.

    >>> import asyncio
    >>> import aio
    >>> async def main():
    ...     aiters = [
    ...         aio.make_iterable_async(range(0,5)),
    ...         aio.make_iterable_async(range(5,10)),
    ...         aio.make_iterable_async(range(10,15)),
    ...     ]
    ...     async for x in aio.merge(*aiters):
    ...         print(x)
    ...
    >>> asyncio.run(main())
    0
    10
    5
    1
    11
    6
    2
    12
    7
    3
    13
    8
    4
    14
    9

    """
    async_iterables = [
        it
        if hasattr(it, '__anext__')
        else make_iterable_async(it)
        for it in iterables
    ]
    tasks = {asyncio.create_task(anext(ait)): ait for ait in async_iterables}
    while tasks:
        finished_tasks, _ = await asyncio.wait(
            tasks.keys(),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in finished_tasks:
            ait = tasks.pop(task)
            try:
                result = task.result()
            except StopAsyncIteration:
                pass
            else:
                yield result
                tasks[asyncio.create_task(anext(ait))] = ait
