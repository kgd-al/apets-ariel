#!/usr/bin/env python3
"""
Run a rerunnable-robot archive (see common/robot_storage.py) on a Unitree Go1 (Edu).

Counterpart of run_hardware.py (Raspberry Pi + robohat): `Go1Wrapper` plays the role of
`RobohatWrapper` and `run_robot` is the same loop (joystick, pause, brain stepping).

Low-level control goes through the python bindings of unitree_legged_sdk (branch `go1`):

    git clone -b go1 https://github.com/unitreerobotics/unitree_legged_sdk
    cd unitree_legged_sdk && mkdir build && cd build
    cmake -DPYTHON_BUILD=TRUE .. && make
    # -> lib/python/{amd64,arm64}/robot_interface*.so

then either `export UNITREE_LEGGED_SDK=/path/to/unitree_legged_sdk` or pass `--sdk-path`.

Before running (see the notes at the end of the answer for sources):
  1. The machine running this script must be on the 192.168.123.x subnet (e.g. 192.168.123.162/24,
     wired), `ping 192.168.123.10` must work.
  2. SUSPEND THE ROBOT (feet in the air).
  3. Put the dog in low-level mode with the remote: L2+A, L2+A, L2+B, L1+L2+Start.
"""

import logging
import os
import platform
import pprint
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Callable, List, Optional, Tuple

import humanize
import numpy as np
import pandas as pd
from mujoco import mj_forward

from ..common import controllers
from ..common.config import BaseConfig, ViewerConfig, AnalysisConfig
from ..common.controllers.ABCpg import ABCpg
from ..common.controllers.abstract import Controller
from ..common.controllers.cpg import RevolveCPG
from ..common.mujoco.state import MjState
from ..common.robot_storage import RerunnableRobot

if __name__ == "__main__":
    # Access configuration in standalone mode
    from ..zoo.config import Arguments as ZooArguments  # noqa: F401
    from ..cpg_rl.types import Config as CPGRLArguments  # noqa: F401
    from ..g_cpg.config import Config as SymmetryArguments  # noqa: F401


@dataclass
class Arguments(BaseConfig, ViewerConfig, AnalysisConfig):
    robot_archive: Annotated[Path, "Path to the rerunnable-robot archive",
                             dict(required=True)] = None

    servo_mapping: Annotated[Optional[str],
                             "Mapping from (mujoco index) -> (Go1 motor) as a comma-separated list."
                             " Motors are either indices in [0,11] or names (FR_0..RL_2) with the order"
                             " FR, FL, RR, RL x (hip, thigh, calf). Prefix with '-' to invert the joint."
                             " Needs to have the same length as the number of actual hinges (<= 12)."
                             " If unknown provide '?' as an argument to default to range(len(hinges))",
                             dict(required=True)] = None

    joint_mode: Annotated[str,
                          "How to interpret actuator commands. 'normalized': ctrl/range in [-1,1] is mapped"
                          " around the standing pose (same convention as run_hardware.py)."
                          " 'absolute': ctrl is directly the joint angle (rad) in the Go1 convention"
                          " (use when the archived model *is* a Go1-like quadruped)"] = "normalized"
    amplitude: Annotated[str,
                         "[normalized mode] Joint excursion (rad) around the standing pose for"
                         " hip,thigh,calf when ctrl/range = +-1"] = "0.3,0.6,0.5"
    stand_pose: Annotated[str, "Standing/reference pose (rad) as hip,thigh,calf (same for all legs)"] \
        = "0.0,0.8,-1.5"
    max_strength: Annotated[float, "Maximal ratio of actuator strength. Multiplies normalized output"] = 1.0

    kp: Annotated[float, "Position gain of the motors"] = 20.0
    kd: Annotated[float, "Damping gain of the motors"] = 2.0
    damping_kd: Annotated[float, "Damping gain used when idle/terminating (kp is 0 then)"] = 1.0
    max_velocity: Annotated[float, "Maximal speed (rad/s) of the joint targets (slew-rate limiter)"] = 8.0
    ramp_duration: Annotated[float, "Time (s) to go from the current pose to the standing pose"] = 3.0
    power_protect: Annotated[int, "Level of Unitree's Safety.PowerProtect (lower = stricter)"] = 1

    go1_ip: Annotated[str, "Address of the Go1 motor controller"] = "192.168.123.10"
    go1_port: Annotated[int, "UDP port of the Go1 motor controller"] = 8007
    local_port: Annotated[int, "Local UDP port"] = 8080
    sdk_path: Annotated[Optional[str],
                        "Path to unitree_legged_sdk (or directly to the folder containing robot_interface*.so)."
                        " Falls back to $UNITREE_LEGGED_SDK"] = None

    joystick: Annotated[bool, "Whether to try and grab hold of a joystick to control the robot's abcpg"] = False
    move: Annotated[bool, "Should the motors be actually used (for debugging purposes, of course)"] = True
    test_hinge_ix: Annotated[int, "Test a single hinge (identified by mujoco index)"] = -1
    test_hinges: Annotated[bool, "Run the hinges test routine instead of using the controller"] = False


def _import_sdk(sdk_path: Optional[str]):
    arch = "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "amd64"
    roots = [p for p in (sdk_path, os.environ.get("UNITREE_LEGGED_SDK")) if p]
    for root in map(Path, roots):
        lib = root / "lib" / "python" / arch
        sys.path.append(str(lib if lib.is_dir() else root))
    try:
        import robot_interface  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Could not import 'robot_interface' (python bindings of unitree_legged_sdk).\n"
            " Build them with `cmake -DPYTHON_BUILD=TRUE ..` and provide --sdk-path"
            " (or $UNITREE_LEGGED_SDK). Searched: " + (", ".join(roots) or "<nothing provided>")) from e
    return robot_interface


def _floats(spec: str, n: int, what: str) -> List[float]:
    try:
        values = [float(x) for x in spec.split(",")]
    except ValueError as e:
        raise ValueError(f"Bad {what} '{spec}': {e}")
    if len(values) != n:
        raise ValueError(f"Bad {what} '{spec}': expected {n} comma-separated values")
    return values


class Go1Wrapper:
    LOWLEVEL = 0xff

    # unitree_legged_sdk ordering: FR, FL, RR, RL x (hip, thigh, calf)
    NAMES = [f"{leg}_{j}" for leg in ("FR", "FL", "RR", "RL") for j in range(3)]

    # Joint limits (rad), from unitree_ros go1.urdf (identical for all four legs)
    LIMITS = [(-0.863, 0.863), (-0.686, 4.501), (-2.818, -0.888)]
    LIMIT_MARGIN = 0.05  # Stay away from mechanical stops

    def __init__(self, args: Arguments, brain: Controller):
        self.args = args
        self.control_period = 1 / args.control_frequency
        self.brain = brain
        self._dt = 0.002  # Same 500Hz loop as unitree's examples
        self._ix_to_mujoco_hinge = {i: a.name for i, a in enumerate(brain.actuators)}
        n = len(self._ix_to_mujoco_hinge)

        if args.joint_mode not in ("normalized", "absolute"):
            raise ValueError(f"Unknown joint mode '{args.joint_mode}'")
        self._stand_type = _floats(args.stand_pose, 3, "stand pose")
        self._amplitude = _floats(args.amplitude, 3, "amplitude")
        self._stand = self._stand_type * 4
        self._lower = [self.LIMITS[m % 3][0] + self.LIMIT_MARGIN for m in range(12)]
        self._upper = [self.LIMITS[m % 3][1] - self.LIMIT_MARGIN for m in range(12)]
        for m, q in enumerate(self._stand):
            if not self._lower[m] <= q <= self._upper[m]:
                raise ValueError(f"Standing pose outside of joint limits for {self.NAMES[m]}: {q}")

        self._motors, self._signs = self._parse_mapping(args.servo_mapping, n)

        if args.verbosity >= 0:
            print("Hinges mapping:")
            for i, m in enumerate(self._motors):
                print(f" ix={i:02}, motor={m:02} ({self.NAMES[m]}){' (inverted)' if self._signs[i] < 0 else ''}:"
                      f" {self._ix_to_mujoco_hinge[i]}")

        sdk = _import_sdk(args.sdk_path)
        self.udp = sdk.UDP(self.LOWLEVEL, args.local_port, args.go1_ip, args.go1_port)
        self.safe = sdk.Safety(sdk.LeggedType.Go1)
        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        self.udp.InitCmdData(self.cmd)

        self._valid = False  # True once we received an actual state
        self._q_des = list(self._stand)  # last commanded targets
        self._initialized, self._started = True, False

    # ------------------------------------------------------------------------------------------
    def _parse_mapping(self, spec: str, n: int) -> Tuple[List[int], List[int]]:
        if n > 12:
            self._hinge_mapping_error(f"found {n} hinges but the Go1 only has 12 motors")
        if spec is None or spec.strip() == "?":
            return list(range(n)), [+1] * n

        motors, signs = [], []
        try:
            for tok in spec.split(","):
                tok = tok.strip()
                signs.append(-1 if tok.startswith("-") else +1)
                tok = tok.lstrip("+-")
                motors.append(self.NAMES.index(tok.upper()) if tok.upper() in self.NAMES else int(tok))
        except Exception as e:
            self._hinge_mapping_error(f"Failed with exception {e}")

        invalid = [m for m in motors if not 0 <= m < 12]
        if invalid:
            self._hinge_mapping_error(f"referencing non-existent motor(s): {invalid}")
        if len(motors) != len(set(motors)):
            self._hinge_mapping_error("duplicate values in mapping")
        if len(motors) != n:
            self._hinge_mapping_error(f"found {n} hinges, provided mapping has {len(motors)} entries")
        return motors, signs

    def _hinge_mapping_error(self, msg):
        raise RuntimeError(f"Bad hinge mapping: {msg}\n Motors: {', '.join(self.NAMES)}")

    @property
    def hinges(self): return len(self._motors)

    def hinge_name(self, i): return self._ix_to_mujoco_hinge[i]

    def motor(self, i): return self._motors[i]

    def q(self) -> List[float]:
        """Measured angle (rad) of all 12 motors"""
        return [self.state.motorState[m].q for m in range(12)]

    def hinge_positions(self) -> List[float]:
        """Measured angle (rad, Go1 convention) of the motors driving each hinge"""
        return [self.state.motorState[m].q for m in self._motors]

    # ------------------------------------------------------------------------------------------
    def _to_q(self, cmds) -> List[float]:
        """Hinge commands -> targets (rad) for all 12 motors. None means 'stay at standing pose'"""
        q = list(self._stand)
        for m, sign, c in zip(self._motors, self._signs, cmds, strict=True):
            if c is None:
                continue
            if self.args.joint_mode == "normalized":
                q[m] = self._stand[m] + sign * float(np.clip(c, -1, 1)) * self._amplitude[m % 3]
            else:
                q[m] = sign * float(c)
            q[m] = min(max(q[m], self._lower[m]), self._upper[m])
        return q

    def _exchange(self, q_des: Optional[List[float]] = None, kp: float = 0., kd: float = 0.):
        """One communication cycle: read the state, then send the command"""
        self.udp.Recv()
        self.udp.GetRecv(self.state)
        if q_des is None:
            q_des = [0.] * 12
        for m in range(12):
            self.cmd.motorCmd[m].q = float(q_des[m])
            self.cmd.motorCmd[m].dq = 0.
            self.cmd.motorCmd[m].Kp = kp
            self.cmd.motorCmd[m].Kd = kd
            self.cmd.motorCmd[m].tau = 0.
        if self._valid:
            self.safe.PowerProtect(self.cmd, self.state, self.args.power_protect)
        self.udp.SetSend(self.cmd)
        self.udp.Send()

    def _loop(self, duration: float, step: Callable[[float], Optional[bool]]):
        """Call step(t) every dt for `duration` seconds (or until it returns False)"""
        start = next_tick = time.perf_counter()
        while (t := time.perf_counter() - start) < duration:
            if step(t) is False:
                break
            next_tick += self._dt
            if (sleep := next_tick - time.perf_counter()) > 0:
                time.sleep(sleep)
            else:  # Late, do not try to catch up
                next_tick = time.perf_counter()

    def _move_to(self, q_to: List[float], duration: float):
        q_from = list(self._q_des) if self._valid else self.q()

        def step(t):
            r = min(t / duration, 1.) if duration > 0 else 1.
            self._q_des = [a * (1 - r) + b * r for a, b in zip(q_from, q_to)]
            self._exchange(self._q_des, self.args.kp, self.args.kd)

        self._loop(duration, step)
        self._q_des = list(q_to)

    def _damp(self, duration: float):
        q = self.q()
        self._loop(duration, lambda t: self._exchange(q, 0., self.args.damping_kd))

    # ------------------------------------------------------------------------------------------
    def start(self):
        if self._started:
            return
        # The controller only answers when it is being talked to: send zero-gain commands until
        # we receive an actual state (all-zero angles = nothing received)
        print(f"Connecting to {self.args.go1_ip}:{self.args.go1_port}... ", end='', flush=True)
        self._loop(3.0, lambda t: self._wait_for_state())
        if not self._valid:
            raise RuntimeError(
                "\nNo state received from the Go1. Check that:\n"
                f" - {self.args.go1_ip} answers to ping (192.168.123.x subnet)\n"
                " - the robot is in low-level mode (L2+A, L2+A, L2+B, L1+L2+Start)")
        print("OK")
        self._q_des = self.q()
        print(f"Moving to standing pose ({self.args.ramp_duration}s)")
        self._move_to(self._stand, self.args.ramp_duration)
        self._started = True
        print("Started:", self._started)

    def _wait_for_state(self):
        self._exchange(None, 0., 0.)
        if any(abs(q) > 1e-9 for q in self.q()):
            self._valid = True
            return False

    def stop(self):
        if self._started:
            self._move_to(self._stand, 1.0)
        self._started = False
        print("Stopped:", not self._started)

    def terminate(self):
        if not self._initialized:
            return
        try:
            self.stop()
        except Exception as e:  # Try hard to at least end up in damping mode
            print(f"Failed to gracefully stop: {e}")
        if self._valid:
            self._damp(0.5)
        self._initialized, self._started = False, False
        print(f"Terminated. Initialized={self._initialized} Started={self._started}")

    def test_command(self, i: int, delta: float) -> float:
        """Command making hinge i move by `delta` around the standing pose, whatever the joint mode/sign"""
        if self.args.joint_mode == "normalized":
            return delta
        return self._signs[i] * (self._stand[self._motors[i]] + delta)

    def set_joints(self, cmds):
        """Counterpart of RobohatWrapper.set_servos: one command per hinge (see Arguments.joint_mode)"""
        self._target = self._to_q(cmds)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self._initialized:
            self.terminate()

    def run(self, fn: Callable[[float], Optional[List[Optional[float]]]]):
        """Call fn(t) every control period; it must return one command per hinge (None to stop)"""
        print("Running")
        slew = self.args.max_velocity * self._dt
        self._target = list(self._stand)
        next_control = 0.

        def step(t):
            nonlocal next_control
            if t >= next_control:
                next_control += self.control_period
                cmds = fn(t)
                if cmds is None:
                    return False
                self.set_joints(cmds)
                print(f"[Running: t={t:.2f}]", end='\r')
            # Slew-rate limiter: at 500Hz, brain steps are too coarse for the motors
            self._q_des = [q + min(max(target - q, -slew), slew)
                           for q, target in zip(self._q_des, self._target)]
            self._exchange(self._q_des, self.args.kp, self.args.kd)

        try:
            self.start()
            self._loop(self.args.duration, step)
            self.stop()
        except KeyboardInterrupt:
            print("\n[!] Force Quit detected (CTRL+C).")
        finally:
            time.sleep(0.1)
            self.terminate()


class Go1Recorder:
    """Light-weight counterpart of HardwarePlotter: stores measured and commanded values to csv"""

    def __init__(self, wrapper: Go1Wrapper, frequency: float):
        self.wrapper = wrapper
        self.frequency = frequency
        self.pos, self.ctrl = [], []

    def step(self, values):
        self.pos.append(self.wrapper.hinge_positions())
        self.ctrl.append([np.nan if v is None else v for v in values])

    def save(self, base_path: Path):
        n = self.wrapper.hinges
        names = [self.wrapper.hinge_name(i) for i in range(n)]
        df = pd.DataFrame(
            np.hstack([np.array(self.pos).reshape(-1, n), np.array(self.ctrl).reshape(-1, n)]),
            columns=[c + "-pos" for c in names] + [c + "-ctrl" for c in names])
        df.insert(0, "Time", [i / self.frequency for i in range(len(df))])
        csv_file = base_path.with_suffix(".go1_hinges.csv")
        df.to_csv(csv_file, index=False)
        print(f"Saved hinge activity to {csv_file}")


def main(args: Arguments) -> int:
    # ==========================================================================
    # Parse command-line arguments
    if args.verbosity <= 0:
        logging_level = logging.WARNING
    elif args.verbosity <= 2:
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, force=True)
    logging.debug("Debug-level logging")

    if args.verbosity >= 2:
        print("Command line-arguments:")
        pprint.PrettyPrinter(indent=2, width=1).pprint(args.__dict__)

    if args.verbosity > 1:
        print("Deduced options:", end='\n\t')
        pprint.pprint(args)

    # ==========================================================================
    # Prepare and launch
    record = RerunnableRobot.load(args.robot_archive)
    args.override_with(record.config, verbose=True, favor_lhs=True)
    print(args)

    # We do need a mujoco simulation and *yes* it is overkill (but lazy!)
    state, model, data = MjState.from_spec(record.mj_spec).unpacked
    mj_forward(model, data)

    brain_class = controllers.get(record.brain[0])
    if brain_class is RevolveCPG:
        brain_class = ABCpg
    brain = brain_class(
        weights=record.brain[2], state=state, name=args.robot_name_prefix, **record.brain[1])

    with Go1Wrapper(args, brain) as wrapper:
        start = time.perf_counter()

        if args.test_hinge_ix >= 0:
            test_single_hinge(args, wrapper, args.test_hinge_ix)
        elif args.test_hinges:
            test_hinges(args, wrapper)
        else:
            run_robot(args, brain, wrapper)

        if args.verbosity >= 1:
            duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
            print(f"Evaluated {args.robot_archive.absolute().resolve()} in {duration} / {args.duration}s")

    return 0


def _sine(t, period): return float(np.sin(2 * np.pi * t / period))


def test_single_hinge(args: Arguments, wrapper: Go1Wrapper, ix):
    n = wrapper.hinges
    amplitude = .5 if args.joint_mode == "normalized" else 0.3

    def runner(t):
        cmds = [None] * n
        cmds[ix] = wrapper.test_command(ix, amplitude * _sine(t, 2.))
        print(f"hinge {ix} ({wrapper.hinge_name(ix)}) -> motor {wrapper.motor(ix)}: {cmds[ix]:.2f}")
        return cmds

    wrapper.run(runner)


def test_hinges(args: Arguments, wrapper: Go1Wrapper):
    """Wiggle each hinge in turn. Use it to check the mapping (and signs!) before running a controller"""
    n = wrapper.hinges
    single_hinge_duration = 2  # seconds
    args.duration = n * single_hinge_duration
    amplitude = .5 if args.joint_mode == "normalized" else 0.3

    def runner(t):
        cmds = [None] * n
        i = min(int(t // single_hinge_duration), n - 1)
        cmds[i] = wrapper.test_command(i, amplitude * _sine(t, single_hinge_duration))
        return cmds

    wrapper.run(runner)


def run_robot(args: Arguments, brain: Controller, wrapper: Go1Wrapper):
    @dataclass
    class RWState:
        time: float = 0

    n = wrapper.hinges
    assert brain.hinges == n

    os.environ["SDL_VIDEODRIVER"] = "dummy"  # no real display needed

    joystick = None
    if args.joystick:
        import pygame
        pygame.display.init()  # satisfies SDL's internal requirement
        pygame.joystick.init()  # only the module you actually need
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            print("Found joystick:", joystick.get_name())
        else:
            print("\nCould not find any connected joystick\n")

    recorder = Go1Recorder(wrapper, args.control_frequency) if args.plot_brain_activity else None

    sorter = None
    if (indices := getattr(brain, "indices", None)) is not None:
        def sorter(array): return [array[i] for i in indices]

    paused = False
    elapsed = 0

    def runner(t):
        nonlocal elapsed, paused

        if joystick is not None:
            pygame.event.pump()
            alpha = joystick.get_axis(0)
            beta = .5 * (joystick.get_axis(5) - joystick.get_axis(2))
            brain.set(alpha=alpha, beta=beta)

            if joystick.get_button(7):
                paused = not paused
            if joystick.get_button(6):
                return None

        if not paused:
            brain(RWState(time=elapsed))
            elapsed += wrapper.control_period

        values = [None] * n
        if args.move:
            for i, (a, r) in enumerate(zip(brain._actuators, brain._ranges)):
                if args.joint_mode == "absolute":
                    values[i] = a.ctrl[0]
                else:
                    values[i] = float(np.clip(a.ctrl[0] / r * args.max_strength, -1, 1))
            print(values)

        if sorter is not None:
            values = sorter(values)

        if recorder is not None:
            recorder.step(values)

        return values

    wrapper.run(runner)

    if recorder is not None:
        recorder.save(args.robot_archive)


if __name__ == "__main__":
    exit(main(Arguments.parse_command_line_arguments("Rerun evolved champions on a Unitree Go1")))
