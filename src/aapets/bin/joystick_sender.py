#!/usr/bin/env python3
"""
Send a joystick's state over UDP to run_go1.py (--listen-to PORT) for untethered control,
and play the robot's status as sounds on this machine.

Run on the machine the joystick is plugged into (needs `pip install pygame`):

    python joystick_sender.py 192.168.12.1 5005

Sounds (--sound tones, the default, needs an audio output here; --sound bell uses the terminal bell):
  - short blip       : the robot script is connected and about to start the motors
  - two rising beeps : the run ended normally
  - three low beeps  : the run failed (the reason is printed)
  - six fast beeps   : the robot has gone silent (script crashed or killed, wifi lost, never started)
  - one mid beep     : the run was interrupted (Ctrl+C on the robot)

The robot side stops the run (and reports a failure) if packets stop coming for --listen-timeout seconds.
Quitting this script (Ctrl+C or the quit button) asks the robot for a clean end of run, and then waits for
its report.

Packets to the robot are JSON datagrams: {"alpha": float, "beta": float, "pause": bool, "quit": bool}.
The robot sends back {"event": "connected|running|alive|success|failure|interrupted", "message": str}.
Axes/buttons default to the ones of run_go1.py's wired mode:
alpha = axis 0, beta = (axis 5 - axis 2) / 2, pause = button 7, quit = button 6.
"""

import argparse
import json
import math
import os
import socket
import sys
import time
from array import array

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# (frequency Hz, duration s)
SOUNDS = dict(
    connected=[(660, .08)],
    success=[(880, .15), (1320, .25)],
    failure=[(220, .45)] * 3,
    lost=[(660, .12), (330, .12)] * 3,
    interrupted=[(440, .25)],
)
TERMINAL = ("success", "failure", "interrupted")
GAP = .08


class Sounds:
    def __init__(self, mode: str):
        self.mode = mode
        self._cache = {}
        if mode == "tones":
            import pygame
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=1)
                self._pygame = pygame
            except pygame.error as e:
                print(f"[sound] no audio output ({e}): falling back to the terminal bell")
                self.mode = "bell"

    @staticmethod
    def _render(pattern, rate=22050, volume=.5):
        samples = array("h")
        for freq, duration in pattern:
            n = int(rate * duration)
            for k in range(n):
                envelope = min(1., k / 200, (n - k) / 200)  # Fade in/out: no clicks
                samples.append(int(32767 * volume * envelope * math.sin(2 * math.pi * freq * k / rate)))
            samples.extend([0] * int(rate * GAP))
        return samples.tobytes()

    def play(self, name: str) -> float:
        """Start playing (non blocking). Returns the duration to wait for it to finish"""
        pattern = SOUNDS[name]
        if self.mode == "none":
            return 0.
        if self.mode == "bell":
            print("\a", end="", flush=True)
            return 0.
        if name not in self._cache:
            self._cache[name] = self._pygame.mixer.Sound(buffer=self._render(pattern))
        self._cache[name].play()
        return sum(d + GAP for _, d in pattern)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("host", help="Address of the robot's computer (e.g. 192.168.12.1 on its hotspot)")
    parser.add_argument("port", type=int, help="Port given to run_go1.py --listen-to")
    parser.add_argument("--rate", type=float, default=50., help="Packets per second")
    parser.add_argument("--sound", choices=["tones", "bell", "none"], default="tones")
    parser.add_argument("--no-answer-after", type=float, default=8.,
                        help="Seconds without any answer from the robot script before warning")
    parser.add_argument("--silent-after", type=float, default=1.5,
                        help="Seconds without a heartbeat while the robot runs before warning")
    parser.add_argument("--starting-silent-after", type=float, default=20.,
                        help="Same, while the robot is starting its motors (connection, ramp to stand)")
    parser.add_argument("--finish-wait", type=float, default=10.,
                        help="After quitting, seconds to wait for the robot's final report")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--axis-alpha", type=int, default=0)
    parser.add_argument("--axis-beta-plus", type=int, default=5)
    parser.add_argument("--axis-beta-minus", type=int, default=2)
    parser.add_argument("--button-pause", type=int, default=7)
    parser.add_argument("--button-quit", type=int, default=6)
    args = parser.parse_args()

    import pygame
    pygame.display.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() <= args.joystick_index:
        sys.exit("No joystick found")
    js = pygame.joystick.Joystick(args.joystick_index)
    print(f"Using joystick '{js.get_name()}': {js.get_numaxes()} axes, {js.get_numbuttons()} buttons")
    for needed, kind, count in [(args.axis_alpha, "axis", js.get_numaxes()),
                                (args.axis_beta_plus, "axis", js.get_numaxes()),
                                (args.axis_beta_minus, "axis", js.get_numaxes()),
                                (args.button_pause, "button", js.get_numbuttons()),
                                (args.button_quit, "button", js.get_numbuttons())]:
        if needed >= count:
            sys.exit(f"This joystick has no {kind} {needed} ({count} available): adjust the --axis/--button options")

    sounds = Sounds(args.sound)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    target = (socket.gethostbyname(args.host), args.port)
    period = 1 / args.rate

    def send(quit_=False):
        pygame.event.pump()
        packet = dict(alpha=js.get_axis(args.axis_alpha),
                      beta=.5 * (js.get_axis(args.axis_beta_plus) - js.get_axis(args.axis_beta_minus)),
                      pause=bool(js.get_button(args.button_pause)),
                      quit=quit_ or bool(js.get_button(args.button_quit)))
        try:
            sock.sendto(json.dumps(packet).encode(), target)
        except OSError:
            pass
        return packet

    def receive():
        events = []
        while True:
            try:
                data, (ip, _) = sock.recvfrom(2048)
            except BlockingIOError:
                return events
            except ConnectionResetError:  # Windows reports ICMP "port unreachable" this way
                continue
            if ip != target[0]:
                continue
            try:
                msg = json.loads(data)
                events.append((str(msg["event"]), str(msg.get("message", ""))))
            except (ValueError, KeyError, TypeError):
                pass

    started = time.monotonic()
    state = {"robot": None, "last_ack": started, "alarmed": False, "done": False}

    def alarm(text):
        if not state["alarmed"]:
            state["alarmed"] = True
            print(f"\n[!] {text}")
            sounds.play("lost")

    def handle(events):
        for event, message in events:
            state["last_ack"] = time.monotonic()
            if state["alarmed"] and event in ("alive", "running", "connected"):
                state["alarmed"] = False
                print("\n[robot answers again]")
            if event == "connected" and state["robot"] is None:
                state["robot"] = "starting"
                print("\nRobot script connected: starting the motors")
                sounds.play("connected")
            elif event == "running":
                state["robot"] = "running"
            elif event in TERMINAL and not state["done"]:
                state["done"] = True
                print(f"\nRobot reports: {event}" + (f" ({message})" if message else ""))
                time.sleep(sounds.play(event))

    def watch():
        if state["done"]:
            return
        now = time.monotonic()
        if state["robot"] is None:
            if now - started > args.no_answer_after:
                alarm("The robot script does not answer (is run_go1.py running with --listen-to?"
                      " Did it fail before connecting?)")
        else:
            limit = args.silent_after if state["robot"] == "running" else args.starting_silent_after
            if now - state["last_ack"] > limit:
                alarm(f"The robot script has gone silent ({now - state['last_ack']:.1f}s)")

    print(f"Sending to {target[0]}:{target[1]} at {args.rate:g} Hz (Ctrl+C or the quit button to end)")
    next_tick = time.perf_counter()
    quitting = False
    try:
        while not state["done"]:
            packet = send()
            handle(receive())
            watch()
            print(f"alpha={packet['alpha']:+.2f} beta={packet['beta']:+.2f}"
                  f" pause={int(packet['pause'])} quit={int(packet['quit'])}"
                  f" robot={state['robot'] or '?'}   ", end="\r")
            if packet["quit"]:
                quitting = True
                break
            next_tick += period
            time.sleep(max(0., next_tick - time.perf_counter()))
    except KeyboardInterrupt:
        quitting = True

    if quitting and not state["done"]:
        print("\nAsking the robot to end the run, waiting for its report (Ctrl+C again to give up)")
        deadline = time.monotonic() + args.finish_wait
        try:
            while not state["done"] and time.monotonic() < deadline:
                send(quit_=True)
                handle(receive())
                time.sleep(period)
        except KeyboardInterrupt:
            pass
        if not state["done"]:
            print("No final report from the robot")
            time.sleep(sounds.play("lost"))


if __name__ == "__main__":
    main()
