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

import json
import logging
import math
import os
import platform
import pprint
import socket
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
                          " (use when the archived model *is* a Go1-like quadruped)."
                          " 'model': ctrl is a mujoco joint position; the Go1 angle is sign*(ctrl-ref) with ref"
                          " the joint's reference (qpos0) in the archived model (for the modified spec with"
                          " recentred ranges). Sign comes from --servo-mapping ('-' prefix)"] = "model"
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
    limit_margin: Annotated[float, "Safety margin (rad) kept from the Go1's joint limits (targets are clipped to them)"] = 0.05
    power_protect: Annotated[int, "Level of Unitree's Safety.PowerProtect (lower = stricter)"] = 1

    go1_ip: Annotated[str, "Address of the Go1 motor controller"] = "192.168.123.10"
    go1_port: Annotated[int, "UDP port of the Go1 motor controller"] = 8007
    local_port: Annotated[int, "Local UDP port"] = 8080
    sdk_path: Annotated[Optional[str],
                        "Path to unitree_legged_sdk (or directly to the folder containing robot_interface*.so)."
                        " Falls back to $UNITREE_LEGGED_SDK"] = None

    sound: Annotated[str, "End-of-run signals (success / failure patterns): 'bell' (terminal bell, heard on"
                          " the machine you ssh from), 'audio' (play tones with aplay on the robot),"
                          " 'both' or 'none'"] = "bell"
    audio_device: Annotated[Optional[str], "ALSA device for --sound audio/both (aplay -D), default if None"] = None

    joystick: Annotated[bool, "Whether to try and grab hold of a (wired) joystick to control the robot's abcpg."
                              " Fails if none is found"] = False
    listen_to: Annotated[Optional[str],
                         "Opt-in untethered control: UDP port (or host:port) on which to listen for the packets"
                         " of joystick_sender.py (running on another machine). Exclusive with --joystick."
                         " Waits for the first packet before starting the motors and stops (with the failure"
                         " signal) if the link is lost"] = None
    listen_from: Annotated[Optional[str],
                           "[--listen-to] Only accept packets from this IP address (otherwise the first"
                           " sender is locked in)"] = None
    listen_timeout: Annotated[float, "[--listen-to] Seconds without packet before the link is declared lost"] = .5
    listen_wait: Annotated[float, "[--listen-to] Seconds to wait for the first packet before giving up"] = 60.
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


_STATE = {"interrupted": False, "remote": None}

SUCCESS_SOUND = [(880, .15), (1320, .25)]          # Two quick rising beeps
FAILURE_SOUND = [(220, .45), (220, .45), (220, .45)]  # Three long low beeps


def _signal(args, pattern):
    """Play a (frequency Hz, duration s) pattern. Never raises: this is just a notification"""
    try:
        mode = getattr(args, "sound", "bell")
        if mode in ("bell", "both"):
            for _, d in pattern:
                print("\a", end="", flush=True)
                time.sleep(d + .1)
        if mode in ("audio", "both"):
            import shutil, subprocess, tempfile, wave
            if shutil.which("aplay") is None:
                print("[sound] aplay not found, no audio")
                return
            rate = 22050
            chunks = []
            for f, d in pattern:
                t = np.arange(int(rate * d)) / rate
                chunks += [.5 * np.sin(2 * np.pi * f * t), np.zeros(int(rate * .1))]
            data = (np.concatenate(chunks) * 32767).astype("<i2").tobytes()
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                with wave.open(tmp.name, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(rate)
                    w.writeframes(data)
                device = ["-D", args.audio_device] if getattr(args, "audio_device", None) else []
                subprocess.run(["aplay", "-q", *device, tmp.name], timeout=20, check=False)
    except Exception as e:
        print(f"[sound] failed: {e}")


class Go1Wrapper:
    LOWLEVEL = 0xff

    # unitree_legged_sdk ordering: FR, FL, RR, RL x (hip, thigh, calf)
    NAMES = [f"{leg}_{j}" for leg in ("FR", "FL", "RR", "RL") for j in range(3)]

    # Joint limits (rad), from unitree_ros go1.urdf (identical for all four legs)
    LIMITS = [(-0.863, 0.863), (-0.686, 4.501), (-2.818, -0.888)]

    def __init__(self, args: Arguments, brain: Controller, refs: Optional[List[float]] = None,
                 ranges: Optional[List[float]] = None):
        self.args = args
        self.control_period = 1 / args.control_frequency
        self.brain = brain
        self._dt = 0.002  # Same 500Hz loop as unitree's examples
        # NB: SymmetricalABCPG already sorts _actuators/_ranges (and thus `actuators`) by position at
        # construction, so hinge i is simply brain.actuators[i] everywhere (no further reordering)
        self._ix_to_mujoco_hinge = {i: a.name for i, a in enumerate(brain.actuators)}
        self._ranges = ranges  # Half-range (rad) of each hinge, in the same order
        n = len(self._ix_to_mujoco_hinge)

        if args.joint_mode not in ("normalized", "absolute", "model"):
            raise ValueError(f"Unknown joint mode '{args.joint_mode}'")
        self._stand_type = _floats(args.stand_pose, 3, "stand pose")
        self._amplitude = _floats(args.amplitude, 3, "amplitude")
        self._stand = self._stand_type * 4
        self._lower = [self.LIMITS[m % 3][0] + args.limit_margin for m in range(12)]
        self._upper = [self.LIMITS[m % 3][1] - args.limit_margin for m in range(12)]
        for m, q in enumerate(self._stand):
            if not self._lower[m] <= q <= self._upper[m]:
                raise ValueError(f"Standing pose outside of joint limits for {self.NAMES[m]}: {q}")

        self._motors, self._signs = self._parse_mapping(args.servo_mapping, n)

        self._refs = refs
        if args.joint_mode == "model":
            if refs is None or len(refs) != n:
                raise ValueError("Joint mode 'model' needs one reference angle per hinge")
            # Mapped motors start from the centre of the recentred range (ctrl=0, from ref);
            # the others hold their measured pose (set in start()). --stand-pose is not used in this mode
            for m, sign, ref in zip(self._motors, self._signs, refs):
                self._stand[m] = sign * (0. - ref)
                if not self._lower[m] <= self._stand[m] <= self._upper[m]:
                    raise ValueError(f"Centre pose outside of joint limits for {self.NAMES[m]}:"
                                     f" {self._stand[m]:.2f} (check sign/mapping)")

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

    def sign(self, i): return self._signs[i]

    def ref(self, i): return self._refs[i]

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
        refs = self._refs or [0.] * len(self._motors)
        for m, sign, ref, c in zip(self._motors, self._signs, refs, cmds, strict=True):
            if c is None:
                continue
            if self.args.joint_mode == "normalized":
                q[m] = self._stand[m] + sign * float(np.clip(c, -1, 1)) * self._amplitude[m % 3]
            elif self.args.joint_mode == "model":
                q[m] = sign * (float(c) - ref)
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
        if self.args.joint_mode == "model":
            # Only the hinges' reference pose is defined: motors without hinge stay where they are
            for m in set(range(12)) - set(self._motors):
                self._stand[m] = min(max(self._q_des[m], self._lower[m]), self._upper[m])
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

    def test_command(self, i: int, fraction: float) -> float:
        """Command moving hinge i to `fraction` (in [-1,1]) of its full range around its centre pose"""
        if self.args.joint_mode == "normalized":
            return fraction
        delta = fraction * (self._ranges[i] if self._ranges is not None else 0.3)
        if self.args.joint_mode == "model":
            return delta  # ctrl=0 is the centre pose, so delta is a ctrl offset (rad)
        return self._signs[i] * (self._stand[self._motors[i]] + delta)

    def check_ranges(self):
        """[model mode] Print the Go1 span of each hinge and flag those clipped by the joint limits"""
        print("Go1 span of each hinge (centre +- spec range):")
        for i, (m, r) in enumerate(zip(self._motors, self._ranges)):
            c = self._stand[m]
            lo, hi = c - r, c + r
            clipped = lo < self._lower[m] or hi > self._upper[m]
            print(f" ix={i:02} {self.NAMES[m]}: [{lo:+.2f}, {hi:+.2f}] rad, allowed"
                  f" [{self._lower[m]:+.2f}, {self._upper[m]:+.2f}]{'  <-- CLIPPED' if clipped else ''}")

    def hinge_targets(self, cmds) -> List[float]:
        """Target angle (rad, Go1 convention, after clipping) of the motor driving each hinge"""
        q = self._to_q(cmds)
        return [q[m] for m in self._motors]

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
            self._target = list(self._stand)
            self._loop(self.args.duration, step)
            self.stop()
        except KeyboardInterrupt:
            _STATE["interrupted"] = True
            print("\n[!] Force Quit detected (CTRL+C).")
        finally:
            time.sleep(0.1)
            self.terminate()


class Go1Plotter:
    """Counterpart of HardwarePlotter: one row per (named) hinge, measured/target angle and spec ctrl"""

    def __init__(self, wrapper: Go1Wrapper, frequency: float):
        self.wrapper = wrapper
        self.frequency = frequency
        self.t, self.pos, self.target, self.ctrl = [], [], [], []

    def step(self, t: float, values):
        """Record one control step. values: the command of each hinge (None = untouched)"""
        self.t.append(t)
        self.pos.append(self.wrapper.hinge_positions())
        self.target.append(self.wrapper.hinge_targets(values))
        self.ctrl.append([np.nan if v is None else v for v in values])

    def _simulation(self, base_path: Path):
        """Optional simulation ground truth (<archive>.brain_activity.csv), converted to Go1 angles"""
        csv_file = base_path.with_suffix(".brain_activity.csv")
        if not csv_file.exists():
            return None
        df = pd.read_csv(csv_file)
        w = self.wrapper
        names = [w.hinge_name(i) for i in range(w.hinges)]
        if any(f"{c}-{k}" not in df.columns for c in names for k in ("pos", "ctrl")):
            print(f"Ignoring {csv_file}: missing hinge columns")
            return None
        t = [i / self.frequency for i in range(len(df))]
        pos, ctrl = {}, {}
        for i, c in enumerate(names):
            ctrl[i] = df[c + "-ctrl"].to_numpy()
            if w.args.joint_mode == "model":  # same conversion as the commands: qpos -> Go1 angle
                pos[i] = w.sign(i) * (df[c + "-pos"].to_numpy() - w.ref(i))
        return t, pos, ctrl

    def plot(self, base_path: Path, tag: str = ""):
        import matplotlib
        matplotlib.use("Agg")  # Headless (the Pi has no display)
        from matplotlib import pyplot as plt

        w, n = self.wrapper, self.wrapper.hinges
        if not self.t:
            print("Nothing recorded, no plot")
            return
        t = np.array(self.t)
        pos, target, ctrl = (np.array(a, dtype=float).reshape(-1, n) for a in (self.pos, self.target, self.ctrl))
        sim = self._simulation(base_path)

        measured_c, target_c, sim_c = "#2a78d6", "#eb6834", "#1baf7a"  # Categorical slots 1-3
        fig, axes = plt.subplots(ncols=2, nrows=n, sharex=True, figsize=(14, 1.9 * n), squeeze=False)
        for i in range(n):
            name = w.hinge_name(i)
            short = name.split("_", 1)[1] if "_" in name else name
            a0, a1 = axes[i]
            a0.plot(t, target[:, i], color=target_c, lw=1.2, ls="--", label="target")
            a0.plot(t, pos[:, i], color=measured_c, lw=1.5, label="measured")
            a1.plot(t, ctrl[:, i], color=measured_c, lw=1.5, label="ctrl")
            a0.set_title(f"{short} -> {w.NAMES[w.motor(i)]}: angle (rad)", fontsize=9, loc="left")
            a1.set_title(f"{short}: ctrl (rad)", fontsize=9, loc="left")
            if sim is not None:
                t_, sim_pos, sim_ctrl = sim
                if i in sim_pos:
                    a0.plot(t_, sim_pos[i], color=sim_c, lw=1.2, label="simulation")
                a1.plot(t_, sim_ctrl[i], color=sim_c, lw=1.2, label="simulation")
            if i == 0:
                a0.legend(loc="upper right", fontsize=8, frameon=False)
                if sim is not None:
                    a1.legend(loc="upper right", fontsize=8, frameon=False)
        for ax in axes.flat:
            ax.grid(alpha=.3)
            ax.margins(x=0)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        fig.tight_layout()

        stem = base_path.with_suffix("")
        pdf_file = stem.parent / f"{stem.name}.hinges{tag}.pdf"
        fig.savefig(pdf_file, bbox_inches="tight")
        plt.close(fig)

        columns = {"Time": t}
        for i in range(n):
            name = w.hinge_name(i)
            columns.update({f"{name}-pos": pos[:, i], f"{name}-target": target[:, i], f"{name}-ctrl": ctrl[:, i]})
        csv_file = stem.parent / f"{stem.name}.go1_hinges{tag}.csv"
        pd.DataFrame(columns).to_csv(csv_file, index=False)
        print(f"Plotted hinge activity to {pdf_file} (and {csv_file.name})")


def _run_recorded(args: Arguments, wrapper: Go1Wrapper, runner, tag: str = ""):
    """wrapper.run(runner) with (optional) recording of the commands and plot at the end"""
    plotter = Go1Plotter(wrapper, args.control_frequency) if args.plot_brain_activity else None

    def recorded(t):
        values = runner(t)
        if plotter is not None and values is not None:
            plotter.step(t, values)
        return values

    try:
        wrapper.run(recorded)
    finally:  # Also plot what happened before a failure
        if plotter is not None:
            try:
                plotter.plot(args.robot_archive, tag)
            except Exception as e:
                print(f"Could not plot hinge activity: {e}")


def _main(args: Arguments) -> int:
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

    refs = _hinge_refs(model, brain) if args.joint_mode == "model" else None
    with Go1Wrapper(args, brain, refs, _hinge_ranges(brain)) as wrapper:
        if args.joint_mode == "model":
            wrapper.check_ranges()
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


def _notify(event: str, message: str = ""):
    remote = _STATE.get("remote")
    if remote is not None:
        remote.notify(event, message, repeat=3)


def main(args: Arguments) -> int:
    try:
        rc = _main(args)
    except Exception as e:
        _notify("failure", f"{type(e).__name__}: {e}")
        _signal(args, FAILURE_SOUND)
        raise
    if _STATE["interrupted"]:  # Ctrl+C is neither a success nor a failure
        _notify("interrupted")
    else:
        _notify("success")
        _signal(args, SUCCESS_SOUND)
    return rc


def _hinge_refs(model, brain) -> List[float]:
    """Reference angle (mujoco `ref`, i.e. qpos0 of hinges) of each actuated joint, in brain.actuators order"""
    import mujoco
    refs = []
    for a in brain._actuators:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a.name)
        jid = model.actuator_trnid[aid, 0]
        refs.append(float(model.qpos0[model.jnt_qposadr[jid]]))
    return refs


def _hinge_ranges(brain) -> List[float]:
    """Half-range (rad, ctrlrange upper bound) of each actuated joint, in brain.actuators order"""
    return [float(r) for r in brain._ranges]


TEST_PERIOD = 5.  # seconds per full-range cycle in the hinge tests


def _sine(t, period): return float(np.sin(2 * np.pi * t / period))


def test_single_hinge(args: Arguments, wrapper: Go1Wrapper, ix):
    n = wrapper.hinges

    def runner(t):
        cmds = [None] * n
        cmds[ix] = wrapper.test_command(ix, _sine(t, TEST_PERIOD))
        print(f"hinge {ix} ({wrapper.hinge_name(ix)}) -> motor {wrapper.motor(ix)}: {cmds[ix]:.2f}")
        return cmds

    _run_recorded(args, wrapper, runner, "-test")


def test_hinges(args: Arguments, wrapper: Go1Wrapper):
    """Wiggle each hinge in turn. Use it to check the mapping (and signs!) before running a controller"""
    n = wrapper.hinges
    single_hinge_duration = TEST_PERIOD  # seconds: one full-range cycle per hinge
    args.duration = n * single_hinge_duration

    def runner(t):
        cmds = [None] * n
        i = min(int(t // single_hinge_duration), n - 1)
        cmds[i] = wrapper.test_command(i, _sine(t - i * single_hinge_duration, single_hinge_duration))
        return cmds

    _run_recorded(args, wrapper, runner, "-test")


class _Edge:
    """Rising edge detector (a held button must toggle the pause only once)"""
    def __init__(self): self._previous = False

    def __call__(self, pressed: bool) -> bool:
        edge, self._previous = bool(pressed) and not self._previous, bool(pressed)
        return edge


class JoystickInput:
    """Wired joystick (pygame). Raises if there is none"""

    def __init__(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # no real display needed
        import pygame
        self._pygame = pygame
        pygame.display.init()  # satisfies SDL's internal requirement
        pygame.joystick.init()  # only the module you actually need
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("--joystick was requested but no joystick was found"
                               " (check `ls /dev/input` and `lsusb`)")
        self._js = pygame.joystick.Joystick(0)
        self._edge = _Edge()
        print("Found joystick:", self._js.get_name())

    def wait(self): pass

    def poll(self) -> dict:
        self._pygame.event.pump()
        js = self._js
        return dict(alpha=float(js.get_axis(0)),
                    beta=float(.5 * (js.get_axis(5) - js.get_axis(2))),
                    toggle_pause=self._edge(js.get_button(7)),
                    quit=bool(js.get_button(6)))


class RemoteInput:
    """Joystick state received over UDP from joystick_sender.py (JSON datagrams:
    {"alpha": float, "beta": float, "pause": bool, "quit": bool}). The first valid sender is locked in.
    Losing the link raises (the run is then ended in a controlled way, with the failure signal)"""

    def __init__(self, listen_to: str, allowed_ip: Optional[str], timeout: float, wait: float):
        host, _, port = listen_to.rpartition(":")
        try:
            port = int(port)
        except ValueError:
            raise ValueError(f"Bad --listen-to '{listen_to}': expected PORT or HOST:PORT")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host or "0.0.0.0", port))
        self._sock.setblocking(False)
        self._allowed, self._peer = allowed_ip, None
        self._timeout, self._wait = timeout, wait
        self._last = None
        self._alpha, self._beta, self._quit, self._toggles = 0., 0., False, 0
        self._edge = _Edge()
        self._ignored = 0
        self._addr, self._heartbeat, self._running = None, 0., False
        print(f"Listening for joystick packets on {host or '0.0.0.0'}:{port}"
              + (f" (only from {allowed_ip})" if allowed_ip else ""))

    def _drain(self):
        while True:
            try:
                data, (ip, port) = self._sock.recvfrom(1024)
            except BlockingIOError:
                return
            try:
                if self._allowed is not None and ip != self._allowed:
                    raise ValueError("unauthorised sender")
                if self._peer is not None and ip != self._peer:
                    raise ValueError("another sender")
                msg = json.loads(data)
                alpha, beta = float(msg["alpha"]), float(msg["beta"])
                if not (math.isfinite(alpha) and math.isfinite(beta)):
                    raise ValueError("not finite")
            except (ValueError, KeyError, TypeError):
                self._ignored += 1
                continue
            if self._peer is None:
                self._peer = ip
                print(f"Locked on joystick sender {ip}")
            self._last = time.monotonic()
            self._addr = (ip, port)
            self._alpha, self._beta = min(max(alpha, -1.), 1.), min(max(beta, -1.), 1.)
            self._toggles += self._edge(msg.get("pause", False))
            self._quit = self._quit or bool(msg.get("quit", False))

    def wait(self):
        """Block until the first valid packet (before any motor is started)"""
        print(f"Waiting for the first packet (up to {self._wait}s)...")
        deadline = time.monotonic() + self._wait
        while self._last is None:
            self._drain()
            if time.monotonic() > deadline:
                raise RuntimeError(f"No joystick packet received within {self._wait}s")
            time.sleep(.01)
        print("Remote control connected")
        self.notify("connected")

    def notify(self, event: str, message: str = "", repeat: int = 1):
        """Tell the sender what happens (it plays the sounds). Best effort, never raises"""
        if self._addr is None:
            return
        payload = json.dumps(dict(event=event, message=message[:300])).encode()
        for k in range(repeat):  # UDP: terminal events are repeated
            try:
                self._sock.sendto(payload, self._addr)
            except OSError:
                return
            if k + 1 < repeat:
                time.sleep(.03)

    def poll(self) -> dict:
        self._drain()
        silence = time.monotonic() - self._last
        if silence > self._timeout:
            raise RuntimeError(f"Remote control link lost ({silence:.1f}s without packet)")
        now = time.monotonic()
        if not self._running:
            self._running = True
            self.notify("running")
        if now - self._heartbeat > .5:  # Lets the sender notice if this script dies
            self._heartbeat = now
            self.notify("alive")
        toggles, self._toggles = self._toggles, 0
        return dict(alpha=self._alpha, beta=self._beta, toggle_pause=toggles % 2 == 1, quit=self._quit)


def run_robot(args: Arguments, brain: Controller, wrapper: Go1Wrapper):
    @dataclass
    class RWState:
        time: float = 0

    n = wrapper.hinges
    assert brain.hinges == n

    if args.joystick and args.listen_to:
        raise ValueError("--joystick and --listen-to are exclusive")

    inputs = None
    if args.joystick:
        inputs = JoystickInput()
    elif args.listen_to:
        inputs = RemoteInput(args.listen_to, args.listen_from, args.listen_timeout, args.listen_wait)
        _STATE["remote"] = inputs
    if inputs is not None:
        inputs.wait()  # Nothing moves until the controls are there

    paused = False
    elapsed = 0

    def runner(t):
        nonlocal elapsed, paused

        if inputs is not None:
            controls = inputs.poll()
            brain.set(alpha=controls["alpha"], beta=controls["beta"])
            if controls["toggle_pause"]:
                paused = not paused
            if controls["quit"]:
                return None

        if not paused:
            brain(RWState(time=elapsed))
            elapsed += wrapper.control_period

        values = [None] * n
        if args.move:
            for i, (a, r) in enumerate(zip(brain._actuators, brain._ranges)):
                if args.joint_mode in ("absolute", "model"):
                    values[i] = float(a.ctrl[0])
                else:
                    values[i] = float(np.clip(a.ctrl[0] / r * args.max_strength, -1, 1))

        return values

    _run_recorded(args, wrapper, runner)


if __name__ == "__main__":
    exit(main(Arguments.parse_command_line_arguments("Rerun evolved champions on a Unitree Go1")))
