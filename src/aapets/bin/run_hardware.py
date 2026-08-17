#!/usr/bin/env python3

# This is mostly a WET copy-paste of rerun.py made to work on the hardware robots
# Mainly strips away irrelevant functionalities (e.g. genome printing) but does provide some
#  testing goodies (no promises yet)

import math
import pickle
import platform

from ..common.controllers.abstract import Controller
if "rpt-rpi" not in platform.platform():
    raise RuntimeError(f"This script is meant to run on an actual robot (with raspberry pi os).\n"
                       f"Expecting *rpt-rpi* not {platform.platform()}")

import time
import logging
import pprint
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Callable, Optional

import humanize
from mujoco import mj_forward

from ..common import controllers
from ..common.config import BaseConfig, ViewerConfig, AnalysisConfig
from ..common.mujoco.state import MjState
from ..common.robot_storage import RerunnableRobot

from robohatlib.Robohat import Robohat # type: ignore
from testlib import TestConfig  # type: ignore

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

    debug: Annotated[bool, "Whether to allow more introspective stuff to run"] = False

    plot: Annotated[bool, "Whether to do any plotting"] = True
    test_hinge_pin: Annotated[int, "Test a single hinge (identified by pin number)"] = -1
    test_hinge_ix: Annotated[int, "Test a single hinge (identified by mujoco index)"] = -1
    test_hinges: Annotated[bool, "Run the test hinges routine instead of using the controller"] = False


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

        self._ix_to_mujoco_hinge = {i: a.name for i, a in enumerate(brain.actuators)}
        self._pins = [i for i in range(32) if self.get_servo_is_connected(i)]
        if args.servo_mapping[0] != "?":
            try:
                servo_mapping = [int(x) for x in args.servo_mapping.split(",")] 
            except Exception as e:
                self._hinge_mapping_error(f"Failed with exception {e}")
            for pin in servo_mapping:
                if pin not in self._pins:
                    self._hinge_mapping_error(f"referencing unconnected pin {pin}")
            if len(servo_mapping) != len(set(servo_mapping)):
                self._hinge_mapping_error("duplicate values in mapping")
            if (n_ := len(servo_mapping)) != (n := len(self._ix_to_mujoco_hinge)):
                self._hinge_mapping_error(f"found {n} hinges, provided mapping has {n_} entries")
        else:
            servo_mapping = list(range(len(self._ix_to_mujoco_hinge)))
        self._pins = [servo_mapping[i] for i in range(len(self._pins))]

        if args.verbosity >= 0:
            print("Hinges mapping:")
            for i, pin in enumerate(self._pins):
                print(f"  ix={i:02}, pin={pin:02}: {self._ix_to_mujoco_hinge[i]}")

        self._initialized, self._started = True, False

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

    def stop(self):
        self.do_buzzer_beep()
        self.stop_servo_drivers()
        self.put_servo_to_sleep()
        self._started = False

    def terminate(self):
        self.do_buzzer_slowwoop()

        self.stop_servo_drivers()
        self.put_servo_to_sleep()
        self.exit_program()
        self._initialized = False

    def set_servos(self, angles):
        _angles = [90.0] * 32
        for pin, a in zip(self._pins, angles, strict=True):
            _angles[pin] = a
        self.set_servo_multiple_angles(_angles)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        if self._initialized:
            self.terminate()
        return self

    def run(self, fn: Callable[[float],list[float]]):
        self.start()
        try:
            start_time = time.perf_counter()
            prev_time = start_time
            while (elapsed_time := time.perf_counter() - start_time) < self.args.duration:
                angles = fn(elapsed_time)
                self.set_servos(angles)

                # print(f"Sleeping for {self.control_period} - {time.perf_counter() - prev_time}")
                time.sleep(self.control_period - time.perf_counter() + prev_time)
                prev_time = time.perf_counter()


        except KeyboardInterrupt:
            print("\n[!] Force Quit detected (CTRL+C).")

        finally:
            self.terminate()



def main(args: Arguments) -> int:
    start = time.perf_counter()

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

    output_prefix = args.robot_archive.with_suffix("")
    plot_ext = args.plot_format

    # We do need a mujoco simulation and *yes* it is overkill (but lazy!)
    state, model, data = MjState.from_spec(record.mj_spec).unpacked
    mj_forward(model, data)

    brain = controllers.get(record.brain[0])(
        weights=record.brain[2], state=state, name=args.robot_name_prefix, **record.brain[1])

    with RobohatWrapper(args, brain) as wrapper:

        if args.test_hinge_pin >= 0:
            candidates = [i for i, pin in enumerate(wrapper._pins) if pin == args.test_hinge_pin]
            assert len(candidates) == 1
            test_single_hinge(args, wrapper, candidates[0])

        elif args.test_hinge_ix >= 0:
            test_single_hinge(args, args.test_hinge_ix, wrapper)

        elif args.test_hinges:
            test_hinges(args, wrapper)

        else:
            run_robot(args, brain, wrapper)

    if args.verbosity >= 1:
        duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
        print(f"Evaluated {args.robot_archive.absolute().resolve()} in {duration} / {state.time}s")

    return 0


def test_single_hinge(args: Arguments, wrapper: RobohatWrapper, ix):
    n = wrapper.hinges
    def runner(t):
        angles = [90] * n
        angles[ix] = 90.0 + (90.0 * math.sin(.5 * t * 2 * math.pi))
        return angles

    wrapper.run(runner)


def test_hinges(args: Arguments, wrapper: RobohatWrapper):
    n = wrapper.hinges
    single_hinge_duration = 2 # seconds
    args.duration = n * single_hinge_duration

    if args.plot:
        data = [[] for _ in range(2*n)]

    def runner(t):
        angles = [90] * n
        angle = 90.0 + (90.0 * math.sin(t * 2 * math.pi / single_hinge_duration))
        i = min(int(t // single_hinge_duration), n-1)
        # print(f"{t=} setting hinge {i} ({wrapper.hinge_name(i)} on pin {wrapper.pin(i)})")
        angles[i] = angle

        if args.plot:
            positions = wrapper.get_servo_multiple_angles()
            for i in range(n):
                data[i+n].append(angles[i])
                data[i].append(positions[wrapper.pin(i)])

        return angles

    wrapper.run(runner)

    if args.plot:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(ncols=2, nrows=n, sharex=True, sharey=True, figsize=(16, 2*n))

        legend = []
        t = [i / args.control_frequency for i in range(len(data[0]))]
        for i in range(n):
            hinge = wrapper.hinge_name(i)
            label = hinge
            if label not in legend:
                legend.append(label)
            else:
                label = "_" + label

            axes[i][0].plot(t, data[i], label=label)
            axes[i][0].set_title(hinge + ": pos")
            axes[i][1].plot(t, data[i+n])
            axes[i][1].set_title(hinge + ": ctrl")

        fig.tight_layout()
        fig.savefig(args.robot_archive.with_suffix(".hinges.pdf"), bbox_inches="tight")
        with open(args.robot_archive.with_suffix(".hinges.pkl"), "wb") as f:
            pickle.dump(data, f)



def run_robot(args: Arguments, brain: Controller, wrapper: RobohatWrapper):
    print("Running robot")
    print(args)
    print("Bye")


if __name__ == "__main__":
    exit(main(Arguments.parse_command_line_arguments("Rerun evolved champions")))
