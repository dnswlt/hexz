"""Performance-related helpers."""

import collections
from contextlib import contextmanager
from functools import wraps
import threading
import time


class Timing(threading.local):
    """Stores accumulated running time in micros and call counts of functions annotated with @timing."""
    def __init__(self):
        self.acc_micros = collections.Counter()
        self.call_counts = collections.Counter()
        self.enabled = True        


_TIMING = Timing()


def disable_perf_stats():
    _TIMING.enabled = False    


def timing(f):
    if not _TIMING.enabled:
        return f

    @wraps(f)
    def wrap(*args, **kw):
        t_start = time.perf_counter_ns()
        result = f(*args, **kw)
        t_end = time.perf_counter_ns()
        name = f.__qualname__
        _TIMING.acc_micros[name] += (t_end - t_start) // 1000
        _TIMING.call_counts[name] += 1
        return result

    return wrap


@contextmanager
def print_time(name):
    """Context manager to print the time a code block took to execute."""
    t_start = time.perf_counter_ns()
    yield
    elapsed = time.perf_counter_ns() - t_start
    print(f"{name} took {int(elapsed/1e6)} ms")


@contextmanager
def timing_ctx(name):
    """Context manager to time a block of code. While @timing can only be used on functions,
    this can be used on any block of code.
    """
    if not _TIMING.enabled:
        yield
        return
    t_start = time.perf_counter_ns()
    yield
    t_end = time.perf_counter_ns()
    _TIMING.acc_micros[name] += (t_end - t_start) // 1000
    _TIMING.call_counts[name] += 1


def clear_perf_stats():
    _TIMING.acc_micros.clear()
    _TIMING.call_counts.clear()


def print_perf_stats():
    if not _TIMING.enabled:
        return
    ms = _TIMING.acc_micros
    ns = _TIMING.call_counts
    width = max(len(k) for k in ms)
    print(f"{'method'.ljust(width)} {'total_sec':>11} {'count':>10} {'ops/s':>10}")
    for k in _TIMING.acc_micros:
        print(
            f"{k.ljust(width)} {ms[k]/1e6:>10.3f}s {ns[k]: 10} {ns[k]/(ms[k]/1e6):>10.1f}"
        )
