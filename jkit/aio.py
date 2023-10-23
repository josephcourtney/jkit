import asyncio
import concurrent.futures
import random
import sys
from collections.abc import Iterable, Coroutine, Generator
from itertools import cycle, islice
from typing import Any, AsyncGenerator, Awaitable

from rich import print

# def desync(f, *args, max_workers=5, **kwargs):
#     async def drain(aiter):
#
#     with asyncio.Runner() as runner:


async def ensure_async(it: Iterable | AsyncGenerator) -> AsyncGenerator:
    """Make an async iterable from a synchronous one by awaiting before each item."""
    if isinstance(it, AsyncGenerator):
        async for item in it:
            yield item
    else:
        for item in it:
            yield item


def ensure_sync(it: AsyncGenerator):
    async def drain_aiter(aiter):
        ret = []
        try:
            async for item in aiter:
                ret.append(item)
        except Exception as e:
            ret.append(e)
        return ret

    if isinstance(it, AsyncGenerator):
        with asyncio.Runner() as runner:
            yield from runner.run(drain_aiter(it))
    else:
        yield from it


async def wobbly(it: Iterable):
    async def _f(e):
        await asyncio.sleep(random.random() * 0.1)
        return e

    awaitables = [_f(e) for e in it]
    g = asyncio.as_completed(awaitables)
    async with asyncio.TaskGroup():
        return [await task for task in g]


with asyncio.Runner() as runner:
    print(runner.run(wobbly([1, 2, 3])))


sys.exit()


def merge(*iterables: Iterable[Iterable]) -> Generator:
    """Merge syncronous iterables into a one by round robin."""
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for nxt in nexts:
                yield nxt()
        except StopIteration:  # noqa: PERF203
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def amerge(*aiters):
    queue = asyncio.Queue(1)
    run_count = len(aiters)
    cancelling = False

    async def drain(aiter):
        nonlocal run_count
        try:
            async for item in aiter:
                await queue.put((False, item))
        except Exception as e:
            if not cancelling:
                await queue.put((True, e))
            else:
                raise
        finally:
            run_count -= 1

    async def merged():
        try:
            while run_count:
                raised, next_item = await queue.get()
                if raised:
                    cancel_tasks()
                    raise next_item
                yield next_item
        finally:
            cancel_tasks()

    def cancel_tasks():
        nonlocal cancelling
        cancelling = True
        for t in tasks:
            t.cancel()

    tasks = []
    for aiter in aiters:
        _aiter = ensure_async(aiter)
        tasks.append(
            asyncio.create_task(drain(_aiter)),
        )
    return merged()


async def raising_aiter():
    for i, e in enumerate("abcdefg"):
        if i == 3:
            msg = "whoopsie"
            raise ValueError(msg)
        yield e


async def main() -> None:
    print()
    async for e in amerge(
        raising_aiter(),
        "D",
        "EF",
    ):
        print(e)


if __name__ == "__main__":
    with asyncio.Runner() as runner:
        runner.run(main())
