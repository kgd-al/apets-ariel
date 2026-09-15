#!/usr/bin/env python3

# This is mostly a WET copy-paste of rerun.py made to work on the hardware robots
# Mainly strips away irrelevant functionalities (e.g. genome printing) but does provide some
#  testing goodies (no promises yet)

import math
import pickle
import platform
import threading

import cv2
import numpy as np
import pandas as pd

from ..common.controllers.ABCpg import ABCpg
from ..common.controllers.abstract import Controller
from ..common.controllers.cpg import RevolveCPG
if "rpt-rpi" not in platform.platform():
    raise RuntimeError(f"This script is meant to run on an actual robot (with raspberry pi os).\n"
                       f"Expecting *rpt-rpi* not {platform.platform()}")

import time
import logging
import pprint
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Callable, Optional, Tuple

import humanize
from mujoco import mj_forward

from ..common import controllers
from ..common.config import BaseConfig, ViewerConfig, AnalysisConfig
from ..common.mujoco.state import MjState
from ..common.robot_storage import RerunnableRobot

from robohatlib.Robohat import Robohat # type: ignore
from testlib import TestConfig  # type: ignore
from robohatlib.hal.datastructure.Color import Color

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
                             "Mapping from (mujoco index) -> (hardware pin) as a comma-separated list." \
                             " Needs to have the same length as the number of actual hinges." \
                             " If unknown provide '?' as an argument to default to range(len(hinges))",
                             dict(required=True)] = None

    max_strength: Annotated[float, "Maximal ratio of actuator strength. Multiplies actual output"] = 1.0

    joystick: Annotated[bool, "Whether to try and grab hold of a joystick to control the robot's abcpg"] = True
    track_ball: Annotated[bool, "Whether to try and follow the ball (hsv specifications below)"] = False

    # Painted ball
    target_hue: Annotated[float, "Hue, in [0, 1], of the target object"] = 0.04748603
    target_saturation: Annotated[float, "Saturation, in [0, 1], of the target object"] = 0.90980392
    target_value: Annotated[float, "Value, in [0, 1], of the target object"] = 0.91960784

    debug: Annotated[bool, "Whether to allow more introspective stuff to run"] = False
    move: Annotated[bool, "Should the servo be actually used (for debugging purposes, of course)"] = True

    plot: Annotated[bool, "Whether to do any plotting"] = True
    test_hinge_pin: Annotated[int, "Test a single hinge (identified by pin number)"] = -1
    test_hinge_ix: Annotated[int, "Test a single hinge (identified by mujoco index)"] = -1
    test_hinges: Annotated[bool, "Run the hinges test routine instead of using the controller"] = False
    test_camera: Annotated[bool, "Run the camera test routine instead of using the controller"] = False
    calibrate_camera: Annotated[bool, "Run a routine to detect the proper HSV range for object detection"] = False


class RobohatWrapper(Robohat):
    def __init__(self, args: Arguments, brain: Controller):
        super().__init__(
            TestConfig.SERVOASSEMBLY_1_CONFIG,
            TestConfig.SERVOASSEMBLY_2_CONFIG,
            TestConfig.TOPBOARD_ID_SWITCH
        )
        self.init(
            TestConfig.SERVOBOARD_1_DATAS_LIST,
            TestConfig.SERVOBOARD_2_DATAS_LIST
        )

        self.args = args
        self.control_period = 1 / args.control_frequency
        self.brain = brain

        self._ix_to_mujoco_hinge = {i: a.name for i, a in enumerate(brain.actuators)}
        self._pins = [i for i in range(32) if self.get_servo_is_connected(i)]
        if args.servo_mapping[0] != "?":
            try:
                servo_mapping, servo_signs = [], []
                for x in args.servo_mapping.split(","):
                    v, s = abs(int(x)), -1 if x[0] == "-" else +1
                    servo_mapping.append(v)
                    servo_signs.append(s)
            except Exception as e:
                self._hinge_mapping_error(f"Failed with exception {e}")

            unconnected = []
            for pin in servo_mapping:
                if pin not in self._pins:
                    unconnected.append(pin)
            if len(unconnected) > 0:
                self._hinge_mapping_error(f"referencing unconnected pin(s):"
                                          f" {','.join(str(p) for p in unconnected)}")

            if len(servo_mapping) != len(set(servo_mapping)):
                self._hinge_mapping_error("duplicate values in mapping")
            if (n_ := len(servo_mapping)) != (n := len(self._ix_to_mujoco_hinge)):
                self._hinge_mapping_error(f"found {n} hinges, provided mapping has {n_} entries")
        else:
            servo_mapping = list(range(len(self._ix_to_mujoco_hinge)))
            servo_signs = [+1 for _ in range(len(self._ix_to_mujoco_hinge))]

        self._pins = [servo_mapping[i] for i in range(len(self._pins))]
        self._signs = servo_signs

        if args.verbosity >= 0:
            print("Hinges mapping:")
            for i, pin in enumerate(self._pins):
                print(f"  ix={i:02}, pin={pin:02}: {self._ix_to_mujoco_hinge[i]}")

        self._initialized, self._started = True, False

        picam = self.get_camera().picam2
        picam.set_controls({
            "AwbEnable": True, "AwbMode": 0,  # 0 = auto
            "AeEnable": False,        # disable auto exposure
            "ExposureTime": 2000,     # microseconds, e.g. 2ms shutter
            "AnalogueGain": 8.0       # compensate for the shorter exposure
        })

        # TEST: Auto-detect light condition (takes time, sad)
        picam.set_controls({"AwbEnable": True})
        time.sleep(1.5)  # let it converge to current lighting
        meta = picam.capture_metadata()
        gains = meta["ColourGains"]
        print("Auto-computed color gains:", gains)
        picam.set_controls({"AwbEnable": False, "ColourGains": gains})

        # Reset camera to get better resolution
        picam.stop()
        print(picam.sensor_modes)
        # config = picam.create_video_configuration(main={"size": (320, 240)})
        config = picam.create_video_configuration(main={"size": (160, 120)})
        picam.configure(config)
        picam.start()

    def _hinge_mapping_error(self, msg):
        raise RuntimeError(f"Bad hinge mapping: {msg}\n"
                           f" Connected pins: {self._pins}")

    @property
    def hinges(self): return len(self._pins)

    def hinge_name(self, i): return self._ix_to_mujoco_hinge[i]
    def pin(self, i): return self._pins[i]

    def start(self):
        self.do_buzzer_beep()
        self.start_servo_drivers()
        self.wakeup_servo()
        self.set_servo_direct_mode(True)
        self._started = True
        print("Started:", self._started)

    def stop(self):
        self.do_buzzer_beep()
        self.stop_servo_drivers()
        self.put_servo_to_sleep()
        self._started = False
        print("Stopped:", not self._started)

    def terminate(self):
        self.do_buzzer_slowwoop()

        self.stop_servo_drivers()
        self.put_servo_to_sleep()
        self.exit_program()
        self._initialized, self._started = False, False
        print(f"Terminated. Initialized={self._initialized} Started={self._started}")

    def set_servos(self, angles):
        _angles = [90.0] * 32
        for pin, sign, a in zip(self._pins, self._signs, angles, strict=True):
            _angles[pin] = a if sign > 0 else 180 - a
        self.set_servo_multiple_angles(_angles)

    def get_frame(self):
        frame = self.get_camera().get_capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        if self._initialized:
            self.stop()

    def run(self, fn: Callable[[float],list[float]]):
        print("Running")
        try:
            self.start()
            start_time = time.perf_counter()
            while (elapsed_time := time.perf_counter() - start_time) < self.args.duration:
                step_start = time.time()
                print(f"[Running: t={elapsed_time}]", end='\r')

                angles = fn(elapsed_time)
                if angles is None:
                    break
                if all(a >= 0 for a in angles):
                    self.set_servos(angles)

                # print(f"Sleeping for {self.control_period} - {time.perf_counter() - prev_time}")
                # time.sleep(self.control_period - time.perf_counter() + prev_time)
                time_until_next_step = self.control_period - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            self.stop()

        except KeyboardInterrupt:
            print("\n[!] Force Quit detected (CTRL+C).")

        except Exception as e:
            raise e

        finally:
            time.sleep(0.5)
            self.terminate()


class HardwarePlotter:
    def __init__(self, wrapper: RobohatWrapper, frequency: float):
        self.wrapper = wrapper
        self.data = [[] for _ in range(2*wrapper.hinges)]
        self.frequency = frequency

    def step(self, angles):
        n = self.wrapper.hinges
        positions = self.wrapper.get_servo_multiple_angles()
        for i in range(n):
            self.data[i+n].append(angles[i])
            self.data[i].append(positions[self.wrapper.pin(i)])

    def plot(self, base_path: Path):
        from matplotlib import pyplot as plt

        n = self.wrapper.hinges
        fig, axes = plt.subplots(ncols=2, nrows=n, sharex=True, sharey=True, figsize=(16, 2*n))

        columns = [self.wrapper.hinge_name(i) for i in range(n)]

        legend = []
        t = [i / self.frequency for i in range(len(self.data[0]))]
        for i in range(n):
            hinge = columns[i]
            label = hinge
            if label not in legend:
                legend.append(label)
            else:
                label = "_" + label

            a0, a1 = axes[i][0], axes[i][1]
            a0.plot(t, self.data[i])
            a1.plot(t, self.data[i+n])
            a0.set_title(hinge + ": pos")
            a1.set_title(hinge + ": ctrl")

        if (ground_truth := base_path.with_suffix(".brain_activity.csv")).exists():
            gt_df = pd.read_csv(ground_truth)
            print(gt_df.columns)
            print({c: gt_df[c].iloc[-1] for c in gt_df.columns})
            t_ = [i / self.frequency for i in range(len(gt_df))]
            def scale(x): return 90 + 2 * 90 * x / math.pi
            for i in range(n):
                hinge = columns[i]
                print(f"Potting {hinge}")
                a0, a1 = axes[i][0], axes[i][1]
                a0.plot(t_, scale(gt_df[hinge + "-pos"]))
                a1.plot(t_, scale(gt_df[hinge + "-ctrl"]))
                a0.set_title(a0.title.get_text() + f" / {hinge} : pos")
                a1.set_title(a1.title.get_text() + f" / {hinge} : ctrl")

        for ax in axes.flat:
            ax.grid()

        fig.tight_layout()

        pdf_file = base_path.with_suffix(".hinges.pdf")
        fig.savefig(pdf_file, bbox_inches="tight")
        pd.DataFrame({
            c: (t if i == 0 else self.data[i-1]) for i, c in
            enumerate(["Time"]+[c + "-pos" for c in columns]+[c + "ctrl" for c in columns])
        }).to_csv(base_path.with_suffix(".hinges.csv"))
        print(f"Plotted hinge activity to {pdf_file} (and .csv too)")


class BallTracker:
    @dataclass
    class GHFilter:
        alpha: float = 0.6
        beta: float = 0.3
        pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
        vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

        max_speed = 1.0
        miss_count = 0
        max_misses_before_reset = 8

        move_threshold: float = 0.15

        ready: bool = False

        def make_ready(self, pos):
            self.pos = np.array(pos, dtype=float)
            self.ready = True

        def predict(self, dt: float):
            if self.ready:
                self.pos += self.vel * dt

            return self.pos.copy()

        def update(self, pos, confidence, dt):
            if not self.ready:
                self.make_ready(pos)
                return self.pos.copy(), True

            z = np.array(pos, dtype=float)
            residual = z - self.pos

            if np.linalg.norm(residual) > self.move_threshold:  # False positive
                self.miss_count += 1
                if self.miss_count >= self.max_misses_before_reset:  # Too many false positives: reset
                    self.pos = z
                    self.vel = np.zeros(2, dtype=float)
                    self.miss_count = 0
                    return self.pos.copy(), True

                return self.pos.copy(), False

            self.miss_count = 0
            g, h = self.alpha * confidence, self.beta * confidence
            self.pos += g * residual
            self.vel += self.vel + h * residual / dt

            speed = np.linalg.norm(self.vel)
            if speed > self.max_speed:
                self.vel = self.vel * (self.max_speed / speed)

            return self.pos.copy(), True

        def probable_edge(self):
            return 1 if self.vel[0] > 0 else -1


    def __init__(self, args: Arguments, brain: ABCpg, wrapper: RobohatWrapper):
        self.args = args
        self.brain = brain
        self.wrapper = wrapper

        self.alpha, self.beta = -1.0, 1.0
        self.hsv_target = (args.target_hue, args.target_saturation, args.target_value)

        self.frames = []
        self.last_sight = 0

        self.gh_filter = self.GHFilter()

    def __call__(self, dt: float):
        # print("gyro", self.wrapper.get_imu_gyro())

        frame = self.wrapper.get_frame()
        # predicted_center = self.gh_filter.predict(dt)
        ball = find_ball(frame, hsv_target=self.hsv_target, confidence=0.5, overlay=True)
        if ball is not None:
            self.last_sight = 0
            center, radius, confidence = ball
            # center, accepted = self.gh_filter.update(center, confidence=confidence, dt=dt)
            # if not accepted:
            #     center = predicted_center

            self.alpha = -np.clip(float(2 * center[0] - 1), -1.0, 1.0)
            close = (radius >= .25 and center[1] > 0.9)
            self.beta = 0.0 if close else 1.0 - .5 * abs(self.alpha)
            self.wrapper.set_led_color(Color.GREEN if close else Color.YELLOW)
        else:
            self.last_sight += dt
        #     center = predicted_center
        #     close = False
            self.wrapper.set_led_color(Color.RED)

        if self.last_sight > 0.5 and abs(self.alpha) < 1: # Wait for half a second
            self.last_sight = 0
            self.alpha = np.sign(self.alpha)
            self.wrapper.set_led_color(Color.PURPLE)
            

        # self.alpha = np.clip(float(2 * center[0] - 1), -1.0, 1.0)
        # self.beta = 0.0 if close else 1.0 - abs(self.alpha)
        # print(f"alpha={self.alpha}, beta={self.beta}")
        self.brain.set(alpha=self.alpha, beta=self.beta)

        with np.printoptions(formatter={'float_kind':"{:.2f}".format}):
            w = frame.shape[1]
            fs = w / 512
            cv2.putText(frame, f"a={self.alpha:.2g}, b={self.beta:.2g}", (0, int(.1 * w)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), 2)
            cv2.putText(frame, f"hg-pos: {self.gh_filter.pos}", (0, int(.15 * w)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), 2)
            cv2.putText(frame, f"hg-vel: {self.gh_filter.vel}", (0, int(.2 * w)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), 2)
        self.frames.append(frame)

    def stop(self):
        writer = cv2.VideoWriter('ball_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                 self.args.control_frequency, self.frames[0].shape[:-1][::-1])
        for frame in self.frames:
            writer.write(frame)
        writer.release()


def to_clean_hsv(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv = cv2.merge([h, s, v_eq])
    return hsv


def hsv_filter(hsv, target_hsv_norm, tolerance=(0.05, 0.3, 0.3)):
    """
    frame: HSV uint8 image
    target_hsv_norm: (h, s, v) each in [0, 1]
    tolerance: (h_tol, s_tol, v_tol) each in [0, 1], applied around target
    """
    h, s, v = target_hsv_norm
    h_tol, s_tol, v_tol = tolerance

    # Rescale target to OpenCV ranges: H in [0,179], S/V in [0,255]
    h_cv = h * 179
    s_cv = s * 255
    v_cv = v * 255
    h_tol_cv = h_tol * 179
    s_tol_cv = s_tol * 255
    v_tol_cv = v_tol * 255

    s_low = np.clip(s_cv - s_tol_cv, 0, 255)
    s_high = np.clip(s_cv + s_tol_cv, 0, 255)
    v_low = np.clip(v_cv - v_tol_cv, 0, 255)
    v_high = np.clip(v_cv + v_tol_cv, 0, 255)

    h_low = h_cv - h_tol_cv
    h_high = h_cv + h_tol_cv

    if h_low < 0 or h_high > 179:
        # Wraps around the seam — split into two ranges and OR them
        h_low_wrapped = h_low % 180
        h_high_wrapped = h_high % 180
        mask1 = cv2.inRange(hsv, np.array([0, s_low, v_low]), np.array([h_high_wrapped, s_high, v_high]))
        mask2 = cv2.inRange(hsv, np.array([h_low_wrapped, s_low, v_low]), np.array([179, s_high, v_high]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])
        mask = cv2.inRange(hsv, lower, upper)

    return mask


def touches_border(contour, w, h, margin=1):
    x, y, cw, ch = cv2.boundingRect(contour)
    return x <= margin or y <= margin or (x + cw) >= w - margin or (y + ch) >= h - margin


def fit_circle_least_squares(contour):
    pts = contour.reshape(-1, 2).astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(sol[2] + cx**2 + cy**2)
    residual = np.mean(np.abs(np.sqrt((x - cx)**2 + (y - cy)**2) - r))
    return cx, cy, r, residual


def blob_confidence(contour, w, h):
    area = cv2.contourArea(contour)
    if area < 20:  # too small, likely noise
        return 0, None

    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius ** 2

    # How much of the enclosing circle is actually filled? (rejects blobby noise/streaks)
    fullness = area / circle_area if circle_area > 0 else 0

    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    if touches_border(contour, w, h):
        cx, cy, radius, residual = fit_circle_least_squares(contour)
        confidence = 1.0 / (1.0 + residual / radius)
        print(cx, cy, radius, residual, residual / radius, confidence)
    else:
        confidence = fullness * circularity  # both close to 1.0 for a real ball

    return confidence, (int(cx), int(cy), int(radius))


def find_ball(frame, hsv_target: float, confidence: float = 0.6, overlay: bool = False) -> Optional[Tuple[Tuple[float, float], float]]:
    w, h = frame.shape[:2][::-1]
    hsv = to_clean_hsv(frame)
    mask = hsv_filter(hsv, hsv_target)

    # Clean up noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        candidates = [blob_confidence(c, w, h) for c in contours]
        candidates = [c for c in candidates if c[0] > confidence]  # tune this threshold

        if candidates:
            confidence, (cx, cy, radius) = max(candidates, key=lambda c: c[0])
            center_px = (int(cx), int(cy))
            radius_px = int(radius)

            center = (cx / frame.shape[1], cy / frame.shape[0])
            radius /= np.min(frame.shape[:2])

            if overlay:
                cv2.circle(frame, center_px, radius_px, (0, 255, 0), 2)   # outline
                cv2.circle(frame, center_px, 5, (0, 0, 255), -1)       # center dot

                cv2.putText(frame, f"c=({center[0]:.2g}, {center[1]:.2g}) r={radius:.2g}",
                            (0, int(.05 * frame.shape[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, frame.shape[0] / 512, (0, 255, 0), 2)

            return center, radius, confidence

    else:
        return None


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
    # record.config.override_with(args, verbose=True)
    print(args)

    # We do need a mujoco simulation and *yes* it is overkill (but lazy!)
    state, model, data = MjState.from_spec(record.mj_spec).unpacked
    mj_forward(model, data)

    brain_class = controllers.get(record.brain[0])
    if brain_class is RevolveCPG:
        brain_class = ABCpg
    brain = brain_class(
        weights=record.brain[2], state=state, name=args.robot_name_prefix, **record.brain[1])

    with RobohatWrapper(args, brain) as wrapper:
        start = time.perf_counter()

        if args.test_camera:
            test_camera(args, wrapper)

        elif args.calibrate_camera:
            calibrate_camera(args, wrapper)

        elif args.test_hinge_pin >= 0:
            candidates = [i for i, pin in enumerate(wrapper._pins) if pin == args.test_hinge_pin]
            assert len(candidates) == 1
            test_single_hinge(args, wrapper, candidates[0])

        elif args.test_hinge_ix >= 0:
            test_single_hinge(args, wrapper, args.test_hinge_ix)

        elif args.test_hinges:
            test_hinges(args, wrapper)

        else:
            run_robot(args, brain, wrapper)

        if args.verbosity >= 1:
            duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
            print(f"Evaluated {args.robot_archive.absolute().resolve()} in {duration} / {args.duration}s")

    return 0


def test_single_hinge(args: Arguments, wrapper: RobohatWrapper, ix):
    n = wrapper.hinges
    def runner(t):
        angles = [90] * n
        angles[ix] = 90.0 + (90.0 * math.sin(.5 * t * 2 * math.pi))
        print(angles)
        return angles

    wrapper.run(runner)


def test_hinges(args: Arguments, wrapper: RobohatWrapper):
    n = wrapper.hinges
    single_hinge_duration = 2 # seconds
    args.duration = n * single_hinge_duration

    if args.plot_brain_activity:
        plotter = HardwarePlotter(wrapper, args.control_frequency)

    def runner(t):
        angles = [90] * n
        angle = 90.0 + (90.0 * math.sin(t * 2 * math.pi / single_hinge_duration))
        i = min(int(t // single_hinge_duration), n-1)
        # print(f"{t=} setting hinge {i} ({wrapper.hinge_name(i)} on pin {wrapper.pin(i)})")
        angles[i] = angle

        if args.plot_brain_activity:
            plotter.step(angles)

        return angles

    wrapper.run(runner)

    if args.plot_brain_activity:
        plotter.plot(args.robot_archive)


playing_fanfare, stop_fanfare = False, False
def play_victory(wrapper: RobohatWrapper):
    global playing_fanfare
    playing_fanfare = True
    stop_fanfare = False
    freqs = [523, 523, 523, 523, 415, 466, 523, 0, 466, 523]
    durations = [167, 167, 167, 500, 750, 250, 250, 125, 125, 2000]
    pauses = [40,  40,  40,  100, 100, 50,  50,  0,   25,  0]
    for freq, duration, pause in zip(freqs, durations, pauses):
        if stop_fanfare:
            break
        wrapper.do_buzzer_freq(freq)
        time.sleep(duration / 1000)
        wrapper.do_buzzer_freq(0)
        time.sleep(pause / 1000)
    playing_fanfare = False


def test_camera(args: Arguments, wrapper: RobohatWrapper):
    import cv2

    n = wrapper.hinges
    hsv_target = (args.target_hue, args.target_saturation, args.target_value)
    frames = []

    global stop_fanfare

    wrapper.turn_led_on()
    wrapper.set_led_color(Color.RED)

    def runner(t):
        angles = [-1] * n
        frame = wrapper.get_frame()
        ball = find_ball(frame, hsv_target=hsv_target, overlay=True)
        if ball is not None:
            center, radius, confidence = ball

            if radius >= .25 and center[1] > 0.9:
                wrapper.set_led_color(Color.GREEN)
                if not playing_fanfare:
                    threading.Thread(target=play_victory, args=(wrapper,), daemon=True).start()
            else:
                wrapper.set_led_color(Color.YELLOW)
                stop_fanfare = True
        else:
            wrapper.set_led_color(Color.RED)
            stop_fanfare = True

        frames.append(frame)

        return angles

    wrapper.run(runner)

    wrapper.set_led_color(Color.PURPLE)

    writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                             args.control_frequency, frames[0].shape[:-1][::-1])
    for frame in frames:
        writer.write(frame)
    writer.release()

def calibrate_camera(args: Arguments, wrapper: RobohatWrapper):
    def snapshot ():
        frame = wrapper.get_frame()
        hsv = to_clean_hsv(frame)
        return frame, hsv

    frame, hsv = snapshot()

    hsv_target = (args.target_hue, args.target_saturation, args.target_value)
    mask = hsv_filter(hsv, hsv_target)
    patch_size = 15

    def on_click(event, x, y, flags, param):
        nonlocal hsv_target, mask
        if event == cv2.EVENT_LBUTTONDOWN:
            half = patch_size // 2
            y0, y1 = max(0, y - half), min(hsv.shape[0], y + half + 1)
            x0, x1 = max(0, x - half), min(hsv.shape[1], x + half + 1)
            patch = hsv[y0:y1, x0:x1].reshape(-1, 3)

            lower = np.maximum(patch.min(axis=0) - [5, 30, 30], [0, 0, 0])
            upper = np.minimum(patch.max(axis=0) + [5, 30, 30], [179, 255, 255])

            hsv_target = .5 * (lower + upper) / np.array([179, 255, 255])
            mask = hsv_filter(hsv, hsv_target)
            

            print(f"Clicked ({x},{y})")
            print("Probably:", hsv_target)

    window_name = "Click on target"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)

    while True:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        combined = np.hstack([frame, mask_bgr])
        cv2.imshow(window_name, combined)

        key = (cv2.waitKey(1) & 0xFF)
        if key == ord('q'):
            print("Quitting")
            break
        elif key == ord('r'):
            print("Taking new snapshot")
            frame, hsv = snapshot()

        time.sleep(0.1)

    cv2.destroyAllWindows()


def run_robot(args: Arguments, brain: Controller, wrapper: RobohatWrapper):
    @dataclass
    class RWState:
        time: float = 0

    n = wrapper.hinges
    assert brain.hinges == n

    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # no real display needed

    joystick = None
    if args.joystick:
        import pygame
        pygame.display.init()   # satisfies SDL's internal requirement
        pygame.joystick.init()  # only the module you actually need
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            print("Found joystick:", joystick.get_name())
        else:
            print("\nCould not find any connected joystick\n")

    if args.track_ball:
        ball_tracker = BallTracker(args, brain, wrapper)

    if args.plot_brain_activity:
        plotter = HardwarePlotter(wrapper, args.control_frequency)

    sorter = None
    if (indices := getattr(brain, "indices", None)) is not None:
        def sorter(array): return [array[i] for i in indices]

    paused = False

    elapsed = 0
    def runner(t):
        nonlocal elapsed, paused

        if args.track_ball:
            ball_tracker(t)

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

        angles = [90] * n
        for i, (a, r) in enumerate(zip(brain._actuators, brain._ranges)):
            ctrl = a.ctrl[0] / r
            ctrl *= args.max_strength
            assert -1 <= ctrl <= 1, f"{ctrl=}"
            if args.move:
                angles[i] = ctrl * 90 + 90

        if sorter is not None:
            angles = sorter(angles)

        if args.plot_brain_activity:
            plotter.step(angles)

        # return [90] * n
        return angles

    wrapper.run(runner)

    if args.plot_brain_activity:
        plotter.plot(args.robot_archive)

    if args.track_ball:
        ball_tracker.stop()


if __name__ == "__main__":
    exit(main(Arguments.parse_command_line_arguments("Rerun evolved champions")))
