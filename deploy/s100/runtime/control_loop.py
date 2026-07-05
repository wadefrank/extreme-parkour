"""Reference 50 Hz control-loop integration.

Production motor watchdogs and emergency-stop handling must also exist below
this Python process.  This loop is the integration boundary, not a hard
real-time motor controller.
"""

import threading
import time
from typing import Optional, Protocol

from .policy_runtime import ControlCommand, ControlMode, PolicyInput, S100PolicyRuntime


class SensorSource(Protocol):
    def read(self) -> PolicyInput:
        ...


class CommandSink(Protocol):
    def write(self, command: ControlCommand) -> None:
        ...


def run_control_loop(
    runtime: S100PolicyRuntime,
    sensors: SensorSource,
    commands: CommandSink,
    stop_event: Optional[threading.Event] = None,
    max_steps: Optional[int] = None,
) -> None:
    """Run until stopped, an E-stop is emitted, or ``max_steps`` is reached."""

    period = 1.0 / runtime.config.control_frequency_hz
    deadline = time.monotonic()
    steps = 0
    while stop_event is None or not stop_event.is_set():
        command = runtime.step(sensors.read())
        commands.write(command)
        steps += 1
        if command.mode == ControlMode.ESTOP:
            break
        if max_steps is not None and steps >= max_steps:
            break
        deadline += period
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            # Do not accumulate an ever-growing lag after an overrun.
            deadline = time.monotonic()
