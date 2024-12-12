import time
import threading
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import format_time

class ProgressLogging(Callback):
    def __init__(self, logger, dt=1):
        self._logger = logger
        self._dt = dt

    def _start(self, dsk):
        self._state = None
        self._start_time = default_timer()
        # Start background thread
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
        ndone = len(s["finished"])
        ntasks = sum(len(s[k]) for k in ["ready", "waiting", "running"]) + ndone
        if ndone < ntasks:
            self._log_progress(ndone / ntasks if ntasks else 0, elapsed)

    def _log_progress(self, frac, elapsed):
        percent = frac * 100
        elapsed = format_time(elapsed)
        msg = f"{percent:.1f}% done in {elapsed}"
        self._logger.info(msg)
