# OS imports
import os
import time
import math
import pprint
import pickle
import numpy as np

# VU Robohat imports
try:
    from robohatlib.Robohat import Robohat
    from robohatlib.Robohat import Color
    from testlib import TestConfig
except ImportError:
    print("CRITICAL ERROR: Failed to import Robohatlib or TestConfig.")
    raise

# ============================================================================ #
# Hardware Configuration
# ============================================================================ #
# Change this number to test different legs! (0 through 7)
TEST_SERVO_PIN = int(os.environ.get("SERVO_PIN", 28))
FILENAME = os.environ.get("FILE", "data.pkl")
CONTROL_FREQUENCY = 20  #Hz
CONTROL_PERIOD = 1 / CONTROL_FREQUENCY  #Hz

print(CONTROL_FREQUENCY, CONTROL_PERIOD)

def prepare():
    print("Initializing Robohat hardware layers...")

    # 1. Initialize the board connections using the University's TestConfig
    robohat = Robohat(
        TestConfig.SERVOASSEMBLY_1_CONFIG,
        TestConfig.SERVOASSEMBLY_2_CONFIG,
        TestConfig.TOPBOARD_ID_SWITCH
    )

    # 2. Initialize the boards
    robohat.init(
        TestConfig.SERVOBOARD_1_DATAS_LIST,
        TestConfig.SERVOBOARD_2_DATAS_LIST
    )

    # 3. Sound the buzzer so we know it didn't crash!
    robohat.do_buzzer_beep()

    # 4. WAKE UP THE MOTORS (The safety switch!)
    robohat.start_servo_drivers()
    robohat.set_servo_direct_mode(True)

    return robohat


def set_one_servo(robohat, servo_pin: int, angle: float) -> None:
    """Move one servo using the multi-servo command path used by Robohat Test.py."""
    angles = [90.0] * 32
    angles[servo_pin] = angle
    robohat.set_servo_multiple_angles(angles)


def get_one_servo(robohat, servo_pin: int) -> float:
    return robohat.get_servo_multiple_angles()[servo_pin]


def test_hinge(robohat, servo_pin: int, frequency: float) -> tuple[list[float], list[float]]:
    robohat.start_servo_drivers()
    robohat.wakeup_servo()
    robohat.set_servo_direct_mode(True)
    set_one_servo(robohat, servo_pin, 90.0)
    time.sleep(.01)

    start_time = time.time()
    last_control = time.perf_counter()
    pos, ctrl = [], []
    try:
        while (elapsed_time := time.time() - start_time) < 5:

            angle = 90.0 + (90.0 * math.sin(elapsed_time * 2 * math.pi * frequency))

            print(f"t={elapsed_time}")
            set_one_servo(robohat, servo_pin, angle)
            pos.append(get_one_servo(robohat, servo_pin))
            ctrl.append(angle)

            print(f"Sleeping for {CONTROL_PERIOD} - {time.perf_counter() - last_control}")
            time.sleep(CONTROL_PERIOD - time.perf_counter() + last_control)
            last_control = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[!] Force Quit detected (CTRL+C).")

    finally:
        print("\nTest complete! Returning limb to neutral...")
        set_one_servo(robohat, servo_pin, 90.0)
        time.sleep(0.5)

        # Safely power down the hardware boards
        robohat.stop_servo_drivers()
        robohat.put_servo_to_sleep()

    return pos, ctrl


def test_multiple_frequencies(robohat, servo_pin, frequencies):
    data = dict()
    for f in frequencies:
        pos, ctrl = test_hinge(robohat, servo_pin, f)
        data[f] = dict(pos=np.array(pos), ctrl=np.array(ctrl))

    with open(FILENAME, "wb") as f:
        pickle.dump(data, f)


# ============================================================================ #
# MAIN EXECUTION LOOP
# ============================================================================ #
def main():
    print("======================================")
    print(f"STARTING LIMB TEST RECORDING (PIN {TEST_SERVO_PIN})")
    print("======================================")

    robohat = prepare()

    time.sleep(1.0)

    try:
        # frequencies = [0.5, 1]
        frequencies = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
        test_multiple_frequencies(robohat, TEST_SERVO_PIN, frequencies)

    except KeyboardInterrupt:
        print("\n[!] Force Quit detected (CTRL+C).")

    finally:
        robohat.do_buzzer_slowwoop()

        # Safely power down the hardware boards
        robohat.stop_servo_drivers()
        robohat.put_servo_to_sleep()
        robohat.exit_program()
        print("Hardware safely powered down.")

if __name__ == "__main__":
    main()
