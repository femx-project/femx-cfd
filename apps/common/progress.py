"""Terminal and machine-readable progress for CFD applications."""

import json
import sys


class ConsoleProgress:
    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.tty = self.stream.isatty()
        self.active = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def stage(self, message):
        self._clear()
        self.stream.write(message + "\n")
        self.stream.flush()

    def __call__(self, event):
        kind = event["type"]
        if kind == "stage":
            self.stage(event["message"])
        elif kind == "solve":
            self._solve(event)
        elif kind == "optimizer":
            self._optimizer(event)
        elif kind == "ensemble" and event["status"] == "begin":
            self._ensemble(event)

    def close(self):
        self._clear()

    def _solve(self, event):
        if not self.tty:
            return
        phase = event["phase"].split("-", 1)[0]
        self.stream.write(
            f"\r\033[K{phase:>7}  {event['step']:>4}/{event['total']:<4}"
        )
        self.stream.flush()
        self.active = True

    def _optimizer(self, event):
        self._clear()
        step = event["iteration"]
        max_itrs = event["max_iterations"]
        obj = event["objective"]
        grad = event["gradient_norm"]
        self.stream.write(
            f"iter {step:>4}/{max_itrs:<4}  "
            f"objective {obj:.6e}  |grad| {grad:.6e}\n"
        )
        self.stream.flush()

    def _ensemble(self, event):
        label = "mean" if event["phase"] == "mean" else "member"
        self.stage(
            f"EnVar {label}: {event['step']}/{event['total']} forward"
        )

    def _clear(self):
        if not self.active:
            return
        if self.tty:
            self.stream.write("\r\033[K")
        else:
            self.stream.write("\n")
        self.stream.flush()
        self.active = False


class NullProgress:
    """Progress sink used by non-root MPI processes."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def stage(self, _message):
        return None

    def __call__(self, _event):
        return None

    def close(self):
        return None


class JsonProgress:
    """Write one JSON object per line for GUI and job-runner consumers."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _default(value):
        item = getattr(value, "item", None)
        if item is None:
            raise TypeError(f"{type(value).__name__} is not JSON serializable")
        return item()

    def _write(self, event):
        self.stream.write(json.dumps(event, default=self._default) + "\n")
        self.stream.flush()

    def stage(self, message):
        self._write({"type": "stage", "message": message})

    def __call__(self, event):
        self._write(dict(event))

    def close(self):
        return None


def console(enabled=True):
    """Return terminal progress on root and a no-op sink elsewhere."""

    return ConsoleProgress() if enabled else NullProgress()
