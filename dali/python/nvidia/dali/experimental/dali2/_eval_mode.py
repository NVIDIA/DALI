import threading
from enum import Enum, auto


class EvalMode(Enum):
    """Enum defining different evaluation modes for DALI2 operations.

    Attributes:
        default:    Default evaluation mode. TBD.
        deferred:   Deferred evaluation mode - operations are evaluated only when their results are
                    needed; error reporting (including input validation) may be delayed until the
                    results are requested.
                    In this mode operations with unused results may be skipped and repeated
                    operations may be merged into one.
        eager:      The evaluation starts immediately. Input validation is immediate.
                    The operations may finish asynchronously.
        sync_cpu:   Synchronous evaluation mode - evaluation on the CPU finishes before the
                    operation returns.
        sync_full:  Fully synchronous evaluation mode - evaluation on all devices finishes before
                    the operation returns.
    """

    default = auto()
    deferred = auto()
    eager = auto()
    sync_cpu = auto()
    sync_full = auto()

    def __enter__(self):
        _tls.eval_mode_stack.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        _tls.eval_mode_stack.pop()

    @staticmethod
    def current() -> "EvalMode":
        return _tls.eval_mode_stack[-1]


_tls = threading.local()
_tls.eval_mode_stack = [EvalMode.default]

__all__ = ["EvalMode"]
