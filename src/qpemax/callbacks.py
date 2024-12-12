import time
import threading
import numpy as np
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import format_time
import psutil
from multiprocessing import Pipe, Process, current_process


class _Tracker(Process):
    """Background process to track memory usage
    of the children of the current process"""

    def __init__(self, dt=1):
        super().__init__()
        self.dt = dt
        self.parent_pid = current_process().pid
        self.parent_conn, self.child_conn = Pipe()

    def shutdown(self):
        if not self.parent_conn.closed:
            self.parent_conn.send("shutdown")
            self.parent_conn.close()
        self.join()

    def _update_pids(self, pid):
        return [self.parent] + [
            p for p in self.parent.children() if p.pid != pid and p.status() != "zombie"
        ]

    def run(self):
        self.parent = psutil.Process(self.parent_pid)
        pid = current_process()
        while True:
            try:
                msg = self.child_conn.recv()
            except KeyboardInterrupt:
                continue
            if msg == "shutdown":
                break
            if msg != "update":
                raise ValueError(f"Unrecognized message {msg}")
            pids = self._update_pids(pid)
            try:
                memory = sum(p.memory_info().rss / 1024 ** 2 for p in pids)
            except Exception:
                memory = np.nan
            self.child_conn.send(memory)
            time.sleep(self.dt)
        self.child_conn.close()


class ProgressLogging(Callback):
    def __init__(self, logger, dt=1):
        self._logger = logger
        self._dt = dt
        self._tracker = _Tracker(dt=dt)

    def _start(self, dsk):
        self._state = None
        self._start_time = default_timer()
        # Start background thread
        self._running = True
        self._timer = threading.Thread(target=self._timer_func)
        self._timer.daemon = True
        self._timer.start()
        # Start memory tracker
        self._tracker.start()

    def _pretask(self, key, dsk, state):
        self._state = state

    def _finish(self, dsk, state, errored):
        self._running = False
        self._timer.join()
        # Shutdown memory tracker
        self._tracker.shutdown()

    def _timer_func(self):
        while self._running:
            elapsed = default_timer() - self._start_time
            self._update(elapsed)
            time.sleep(self._dt)

    def _update(self, elapsed):
        s = self._state
        if s is None:
            return
        self._tracker.parent_conn.send("update")
        memory = self._tracker.parent_conn.recv()
        print(memory)
        ndone = len(s["finished"])
        ntasks = sum(len(s[k]) for k in ["ready", "waiting", "running"]) + ndone
        if ndone < ntasks:
            self._log_progress(ndone / ntasks if ntasks else 0, elapsed, memory)

    def _log_progress(self, frac, elapsed, memory):
        percent = frac * 100
        elapsed = format_time(elapsed)
        msg = f"{percent:.1f}% done in {elapsed}, memory usage: {memory:.1f} MB"
        self._logger.info(msg)
