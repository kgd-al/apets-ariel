import atexit
import ctypes
import pprint
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import glfw
import mujoco
import numpy as np
import yaml
from mujoco import mj_step, mjtCamera
from mujoco.viewer import KeyCallbackType, _InternalLoaderType, _reload, _physics_loop, Handle, _MJPYTHON, _Simulate

from .state import MjState
from ..config import ViewerConfig


def interactive_viewer(model, data, args: ViewerConfig):
    mujoco.viewer.launch(model, data)


def passive_viewer(state: MjState, args: ViewerConfig,
                   overlays=None, viewer_ready_callback=None,
                   debug=True):
    paused = not args.auto_start
    overlays = overlays or []

    state, model, data = state.unpacked

    def callback(key: int):
        nonlocal paused
        if key == 32:
            paused = not paused

    with mujoco.viewer.launch_passive(model, data, key_callback=callback) as viewer:
        viewer.verbosity = args.verbosity

        if not debug:
            glfw.set_key_callback(viewer.glfw_window, callback)
            glfw.set_mouse_button_callback(viewer.glfw_window, None)

        match args.camera:
            case None:
                pass

            case "tracking":
                viewer.cam.type = mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = model.body("apet1_world").id

            case _:
                viewer.cam.fixedcamid = model.camera(args.camera).id
                viewer.cam.type = mjtCamera.mjCAMERA_FIXED

        if args.settings_restore:
            restore_persistent_settings(viewer)

        def closing(_w):
            save_persistent_settings(viewer)

        if args.settings_save:
            glfw.set_window_close_callback(viewer.glfw_window, closing)

        if viewer_ready_callback is not None:
            viewer_ready_callback(viewer)

        def maybe_pause():
            while paused and not glfw.window_should_close(viewer.glfw_window):
                viewer.sync()
                time.sleep(.1)

        total_time = 0

        for overlay in overlays:
            overlay.start(viewer, state)

        maybe_pause()

        while viewer.is_running() and total_time < args.duration:
            maybe_pause()

            step_start = time.time()

            mj_step(model, data)
            total_time += model.opt.timestep

            for overlay in overlays:
                overlay.render(viewer, state)
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        if not args.auto_quit:
            paused = True

        for overlay in overlays:
            overlay.stop(viewer, state)

        maybe_pause()


def persistent_settings_storage():
    storage = Path.joinpath(Path.home(), ".config/mujoco/viewer.yaml")
    storage.parent.mkdir(parents=True, exist_ok=True)
    return storage


def save_persistent_settings(viewer):
    window = viewer.glfw_window
    pos = glfw.get_window_pos(window)
    size = glfw.get_window_size(window)

    sim = viewer._get_sim()

    def maybe_to_list(_attr):
        return _attr.tolist() if isinstance(_attr, np.ndarray) else _attr

    def to_yaml(obj, attrs):
        return {
            k: maybe_to_list(getattr(obj, k)) for k in attrs
        }

    viewer_options = to_yaml(viewer.opt, [
        "actuatorgroup", "bvh_depth", "flags", "flexgroup", "flex_layer",
        "frame", "geomgroup", "jointgroup", "label", "sitegroup", "skingroup",
        "tendongroup",
    ])
    scene_options = to_yaml(viewer.user_scn, ["flags"])

    with open(persistent_settings_storage(), "w") as f:
        data = dict(
            pos=pos, size=size,
            ui0=sim.ui0_enable, ui1=sim.ui1_enable,
            viewer_options=viewer_options,
            scene_options=scene_options
        )
        if viewer.verbosity > 1:
            print("[Settings] Saving", pprint.pformat(data))
        yaml.safe_dump(data, f)


def restore_persistent_settings(viewer):
    try:
        window = viewer.glfw_window
        sim = viewer._get_sim()

        with open(persistent_settings_storage(), "r") as f:
            data = yaml.safe_load(f)
            if viewer.verbosity > 1:
                print("[Settings] Restoring", pprint.pformat(data))

            glfw.set_window_pos(window, *data.get("pos"))
            glfw.set_window_size(window, *data.get("size"))
            sim.ui0_enable = data.get("ui0", True)
            sim.ui1_enable = data.get("ui1", True)

            def maybe_numpy(_attr):
                return np.array(_attr) if isinstance(_attr, list) else _attr

            def from_yaml(obj, name):
                if (sub_data := data.get(name)) is not None:
                    for k, v in sub_data.items():
                        setattr(obj, k, maybe_numpy(v))

            from_yaml(viewer.opt, "viewer_options")
            from_yaml(viewer.user_scn, "scene_options")

    except IOError:
        print("No existing persistent settings")

    except AttributeError as e:
        print(f"Error while reading settings:\n{e}")


# /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\
# TODO: Ugly monkey patching to get access to the glfw window. Likely to be very brittle
# Only relevant modification is that the handle is created on the side thread and stores
#  the current glfw context (otherwise said context is illegal)


def _launch_internal(
        model: Optional[mujoco.MjModel] = None,
        data: Optional[mujoco.MjData] = None,
        *,
        run_physics_thread: bool,
        loader: Optional[_InternalLoaderType] = None,
        handle_return: Optional['queue.Queue[Handle]'] = None,
        key_callback: Optional[KeyCallbackType] = None,
        show_left_ui: bool = True,
        show_right_ui: bool = True,
) -> None:
    """Internal API, so that the public API has more readable type annotations."""
    if model is None and data is not None:
        raise ValueError('mjData is specified but mjModel is not')
    elif callable(model) and data is not None:
        raise ValueError(
            'mjData should not be specified when an mjModel loader is used'
        )
    elif loader is not None and model is not None:
        raise ValueError('model and loader are both specified')
    elif run_physics_thread and handle_return is not None:
        raise ValueError('run_physics_thread and handle_return are both specified')

    if loader is None and model is not None:

        def _loader(m=model, d=data) -> Tuple[mujoco.MjModel, mujoco.MjData]:
            if d is None:
                d = mujoco.MjData(m)
            return m, d

        loader = _loader

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    if model and not run_physics_thread:
        user_scn = mujoco.MjvScene(model, _Simulate.MAX_GEOM)
    else:
        user_scn = None

    simulate = _Simulate(
        cam, opt, pert, user_scn, run_physics_thread, key_callback
    )
    simulate.ui0_enable = show_left_ui
    simulate.ui1_enable = show_right_ui

    # Initialize GLFW if not using mjpython.
    if _MJPYTHON is None:
        if not glfw.init():
            raise mujoco.FatalError('could not initialize GLFW')
        atexit.register(glfw.terminate)

    notify_loaded = None
    if handle_return:
        handle = Handle(simulate, cam, opt, pert, user_scn)
        handle.glfw_window = glfw.get_current_context()
        notify_loaded = lambda: handle_return.put_nowait(handle)
        # notify_loaded = lambda: handle_return.put_nowait(
        #     Handle(simulate, cam, opt, pert, user_scn)
        # )

    if run_physics_thread:
        side_thread = threading.Thread(
            target=_physics_loop, args=(simulate, loader)
        )
    else:
        side_thread = threading.Thread(
            target=_reload, args=(simulate, loader, notify_loaded)
        )

    def make_exit(simulate):
        def exit_simulate():
            simulate.exit()

        return exit_simulate

    exit_simulate = make_exit(simulate)
    atexit.register(exit_simulate)

    side_thread.start()
    simulate.render_loop()
    atexit.unregister(exit_simulate)
    side_thread.join()
    simulate.destroy()


mujoco.viewer._launch_internal = _launch_internal
