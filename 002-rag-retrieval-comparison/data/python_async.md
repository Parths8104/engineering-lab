# Python Async and Concurrency

## The asyncio module

Python's `asyncio` provides infrastructure for writing single-threaded concurrent code using coroutines. Coroutines are declared with `async def` and awaited with `await`. The event loop schedules them cooperatively — only one coroutine executes Python at a time, but the loop can switch between coroutines at every `await` point.

## When asyncio helps

Asyncio is a strong fit for I/O-bound workloads with high concurrency: HTTP calls, database queries, WebSocket connections. Because a single event loop can juggle thousands of coroutines with minimal overhead, you can handle massive fan-out from one process. Compare that to threads, which cost around 8 MB of stack per thread on most systems.

## When asyncio hurts

For CPU-bound work, asyncio provides no speedup — the Global Interpreter Lock (GIL) means only one thread executes Python bytecode at a time, and coroutines share that single thread. Use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor` for CPU work.

## The `asyncio.gather` primitive

`asyncio.gather(*coros)` runs multiple coroutines concurrently and returns their results as a list preserving input order. If any coroutine raises, `gather` cancels the others by default and re-raises. Pass `return_exceptions=True` to collect exceptions in the result list instead.
