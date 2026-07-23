"""Terminal and machine-readable progress for CFD case runs."""

import json
import sys
from typing import Optional, TextIO

from femx_cfd._typing import Dynamic, ProgressEvent


class ConsoleProgress:
    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self.stream = stream or sys.stderr
        self.tty = self.stream.isatty()
        self.active = False
        self.member_states: dict[tuple[str, str], ProgressEvent] = {}
        self.member_lines = 0

    def __enter__(self) -> "ConsoleProgress":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def stage(self, message: str) -> None:
        self._clear()
        self.member_states.clear()
        self.stream.write(message + "\n")
        self.stream.flush()

    def __call__(self, event: ProgressEvent) -> None:
        kind = event["type"]
        if kind == "stage":
            self.stage(event["message"])
        elif kind == "solve":
            self._solve(event)
        elif kind == "optimizer":
            self._optimizer(event)
        elif kind == "ensemble" and event["status"] == "begin":
            self._ensemble(event)
        elif kind == "ensemble_solve":
            self._ensemble_solve(event)

    def close(self) -> None:
        self._clear()

    def _solve(self, event: ProgressEvent) -> None:
        phase = event["phase"].split("-", 1)[0]
        # MPI launchers can retain a partial output line.  Terminate the line
        # so it is forwarded immediately, then return to it for the next
        # update.  A terminal still shows only the current solver step.
        self.stream.write(
            f"\r\033[K{phase:>7}  {event['step']:>4}/{event['total']:<4}\n\033[1A"
        )
        self.stream.flush()
        self.active = True

    def _optimizer(self, event: ProgressEvent) -> None:
        self._clear()
        step = event["iteration"]
        max_itrs = event["max_iterations"]
        obj = event["objective"]
        grad = event["gradient_norm"]
        if event.get("iteration_kind") == "evaluation":
            prefix = f"eval {step:>4}       "
        else:
            prefix = f"iter {step:>4}/{max_itrs:<4}  "
        self.stream.write(f"{prefix}objective {obj:.6e}  |grad| {grad:.6e}\n")
        self.stream.flush()

    def _ensemble(self, event: ProgressEvent) -> None:
        label = "mean" if event["phase"] == "mean" else "member"
        self.stage(f"EnVar {label}: {event['step']}/{event['total']} forward")

    def _ensemble_solve(self, event: ProgressEvent) -> None:
        stage = event["stage"]
        member = event["member"]
        if not self.tty:
            self.stream.write(
                f"{stage:>11}  {member:<28} {event['phase']:>7}  "
                f"{event['step']:>4}/{event['total']:<4}\n"
            )
            self.stream.flush()
            return
        self.member_states[(stage, member)] = event
        self._render_member_states()

    def _render_member_states(self) -> None:
        if self.member_lines:
            self.stream.write(f"\033[{self.member_lines}A")
        for event in self.member_states.values():
            self.stream.write(
                f"\r\033[K{event['stage']:>11}  {event['member']:<28} "
                f"{event['phase']:>7}  {event['step']:>4}/{event['total']:<4}\n"
            )
        self.stream.flush()
        self.member_lines = len(self.member_states)

    def _clear(self) -> None:
        if self.member_lines:
            self.stream.write(f"\033[{self.member_lines}A")
            for _ in range(self.member_lines):
                self.stream.write("\r\033[K\n")
            self.stream.write(f"\033[{self.member_lines}A")
            self.stream.flush()
            self.member_lines = 0
        if not self.active:
            return
        self.stream.write("\r\033[K")
        self.stream.flush()
        self.active = False


class NullProgress:
    """Progress sink used by non-root MPI processes."""

    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def stage(self, _message: str) -> None:
        return None

    def __call__(self, _event: ProgressEvent) -> None:
        return None

    def close(self) -> None:
        return None


class JsonProgress:
    """Write one JSON object per line for GUI and job-runner consumers."""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self.stream = stream or sys.stdout

    def __enter__(self) -> "JsonProgress":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _default(value: Dynamic) -> Dynamic:
        tolist = getattr(value, "tolist", None)
        if tolist is not None:
            return tolist()
        item = getattr(value, "item", None)
        if item is None:
            raise TypeError(f"{type(value).__name__} is not JSON serializable")
        return item()

    def _write(self, event: ProgressEvent) -> None:
        self.stream.write(json.dumps(event, default=self._default) + "\n")
        self.stream.flush()

    def stage(self, message: str) -> None:
        self._write({"type": "stage", "message": message})

    def __call__(self, event: ProgressEvent) -> None:
        self._write(dict(event))

    def close(self) -> None:
        return None
