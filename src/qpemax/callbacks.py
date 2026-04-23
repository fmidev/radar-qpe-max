# builtin
import os
import time
import threading
from timeit import default_timer

# pypi
import psutil
from dask.callbacks import Callback
from dask.utils import format_time


class _MemoryTracker:
    """Track RSS memory of the current process plus its children.

    Runs in-process (no subprocess), so it is safe under any multiprocessing
    start method. The previous ``Process``-based implementation failed under
    Python 3.14's default ``forkserver`` start method when the importing
    module (e.g. Airflow's ``/tmp/script.py``) is not a safely importable
    ``__main__``.
    """

    def __init__(self):
        self._parent = psutil.Process(os.getpid())

    def sample(self) -> float:
        try:
            procs = [self._parent] + [
                p for p in self._parent.children(recursive=True)
                if p.status() != "zombie"
            ]
            return sum(p.memory_info().rss / 1024 ** 2 for p in procs)
        except Exception:
            return float("nan")


class ProgressLogging(Callback):
    def __init__(self, logger, dt=1):
        self._logger = logger
        self._dt = dt
        self._tracker = _MemoryTracker()

    def _start(self, dsk):
        self._state = None
        self._start_time = default_timer()
        self._running = True
        self._timer = threading.Thread(target=self._timer_func)
        self._timer.daemon = True
        self._timer.start()

    def _pretask(self, key, dsk, state):
        self._state = state

    def _finish(self, dsk, state, errored):
        self._running = False
        self._timer.join()

    def _timer_func(self):
        while self._running:
            elapsed = default_timer() - self._start_time
            self._update(elapsed)
            time.sleep(self._dt)

    def _update(self, elapsed):
        s = self._state
        if s is None:
            return
        mem = self._tracker.sample()
        ndone = len(s["finished"])
        todo_status = ["ready", "waiting", "running"]
        ntasks = sum(len(s[k]) for k in todo_status) + ndone
        if ndone < ntasks:
            self._log_progress(ndone / ntasks if ntasks else 0, elapsed, mem)

    def _log_progress(self, frac, elapsed, memory):
        percent = frac * 100
        elapsed = format_time(elapsed)
        msg = f"{percent:.1f}% done in {elapsed}, mem: {memory:.1f} MB"
        self._logger.info(msg)
