from typing import List

import glfw
import numpy as np
from mujoco import MjSpec, MjsCamera, mjtCamLight, mju_euler2Quat, mjtEq, mjtObj, mju_rotVecQuat, mju_mulQuat, \
    MjContact, mju_subFrom3, mjNEQDATA, mju_mulMatVec3, mju_mulMatTVec3, mjtTrn, mjtGeom, mjtSensor
from mujoco.viewer import Handle
from numpy import ndarray
from robot_descriptions import allegro_hand_mj_description as robot_hand

from aapets.fetch.controllers.fetcher import FetcherCPG
from .base import GenericFetchDynamics
from ..overlay import FetchOverlay
from ..types import InteractionMode, Config, Buttons, Keys, Constraints
from ...common.mujoco.state import MjState


class FetchDynamics(GenericFetchDynamics):
    def __init__(self,
                 state: MjState,
                 overlay: FetchOverlay,
                 robot: str, ball: str, human: str,
                 brain: FetcherCPG):

        super().__init__(
            state, InteractionMode.HUMAN, overlay,
            robot, ball, human, brain
        )

        self.state = state

        self.__constraints = {c: state.model.equality(c).id for c in Constraints}

        self.__throw_timer = None

        self.__mocap = self.state.model.body("mocap-hand").mocapid[0]
        self.__hand = self.state.data.body("palm")
        self.__core = self.state.model.geom(f"{robot.split('_')[0]}_core")

        self.__previous_mouse_pos = None
        self.__cursor_visible = True

        self.mouth = (self.state.data.actuator("mouth"), self.state.data.sensor("mouth"))

    @classmethod
    def adjust_camera(cls, specs: MjSpec, config: Config):
        camera: MjsCamera = specs.camera(config.camera)

        camera.mode = mjtCamLight.mjCAMLIGHT_TARGETBODY
        camera.targetbody = f"{config.robot_name_prefix}1_world"

        camera.orthographic = False
        camera.fovy = 75

        camera.pos = (config.arena_extent, 0, config.human_height)

    @classmethod
    def adjust_world(cls, specs: MjSpec, config: Config):
        super().adjust_world(specs, config)

        offset = .2 * config.human_height
        pos = (.8 * config.arena_extent - offset, offset, config.human_height)
        mocap = specs.worldbody.add_body(name="mocap-hand", pos=pos, mocap=True)
        # mocap_site = mocap.add_site(name="mocap-site")

        frame = mocap.add_frame()
        mju_euler2Quat(frame.quat, np.array([0, -np.pi / 2, 0]), "xyz")

        hand_specs: MjSpec = MjSpec.from_file(robot_hand.MJCF_PATH_RIGHT)
        specs.attach(hand_specs, frame=frame)

        hand = specs.body("palm")
        hand_site = hand.add_site(
            name="hand-site", pos=(.02, 0, 0)
        )

        robot_name = f"{config.robot_name_prefix}1"
        # robot = specs.body(f"{robot_name}_world")
        cls.add_mouth(specs, robot_name)

        ball = specs.body("ball")
        ball_site = ball.add_site(name="ball-site")

        specs.add_equality(
            name=Constraints.HAND_BALL,
            type=mjtEq.mjEQ_CONNECT,
            objtype=mjtObj.mjOBJ_SITE,
            active=False,
            name1=hand_site.name, name2=ball_site.name
        )

        # specs.add_equality(
        #     name=Constraints.ROBOT_BALL,
        #     type=mjtEq.mjEQ_CONNECT,
        #     objtype=mjtObj.mjOBJ_BODY,
        #     active=False,
        #     name1=robot.name, name2=ball.name
        # )

    @classmethod
    def add_mouth(cls, specs: MjSpec, robot_name: str):
        core_size = specs.geom(f"{robot_name}_core").size
        assert core_size[0] == core_size[1] == core_size[2]
        core_size = core_size[0]

        mouth = specs.body(f"{robot_name}_world").add_body(
            name="mouth",
            pos=(np.sqrt(2) * core_size, 0, -.5 * core_size)
        )
        mouth.add_geom(
            type=mjtGeom.mjGEOM_BOX,
            mass=.001,
            size=(.001, .01, .01)
        )

        mouth_actuator = specs.add_actuator(
            name="mouth",
            target="mouth",
            trntype=mjtTrn.mjTRN_BODY,
            ctrlrange=[0, 1],
        )
        mouth_actuator.set_to_adhesion(gain=50)

        specs.add_sensor(
            name="mouth",
            type=mjtSensor.mjSENS_CONTACT,
            objtype=mjtObj.mjOBJ_BODY,
            objname=mouth.name,
            intprm=[1, 0, 1]
        )

    # -----
    # - GLFW stuff
    # --

    def on_viewer_ready(self, viewer: Handle):
        super().on_viewer_ready(viewer)
        self.__previous_mouse_pos = self._mouse_pos()
        # self.__hide_cursor()
        # print("[kgd-debug] Not hiding cursor")

    def __hide_cursor(self):
        window = self.viewer.glfw_window
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        if glfw.raw_mouse_motion_supported():
            glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, True)
            self.__cursor_visible = False
        else:
            self.__show_cursor()

    def __show_cursor(self):
        glfw.set_input_mode(self.viewer.glfw_window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        self.__cursor_visible = True

    def __switch_cursor(self):
        print(f"[kgd-debug] switching cursor state")
        if self.__cursor_visible:
            print(f"[kgd-debug] hiding")
            self.__hide_cursor()
        else:
            print(f"[kgd-debug] showing")
            self.__show_cursor()
        print(f"[kgd-debug] cursor visible", self.__cursor_visible)

    # -----
    # - Visuals
    # --

    def __update_overlay(self):
        self.overlay.throw_data = self.__hand_throw_norm() * self.__hand_throw_strength() ** .125

    # -----
    # - World dynamics
    # --

    def _step(self, state: MjState):
        super()._step(state)
        if self.__throw_timer is not None:
            self.__update_overlay()

        print(self.mouth)

        # if not self.__is_constraint_active(Constraints.ROBOT_BALL):
        #     if (collision := self.__robot_ball_collisions()) is not None:
        #         self.__robot_grab_ball(*collision)

    # -----
    # - Mujoco accessors
    # --

    def __is_constraint_active(self, c: Constraints):
        return self.state.data.eq_active[self.__constraints[c]]

    def __set_constraint_active(self, c: Constraints, active: bool):
        self.state.data.eq_active[self.__constraints[c]] = active

    # -----
    # - Interactions: ball & hand
    # --

    def __hand_grab_ball(self):
        self.__set_constraint_active(Constraints.ROBOT_BALL, False)
        self.__set_constraint_active(Constraints.HAND_BALL, True)

    def __hand_prepare_throw_ball(self):
        self.__throw_timer = self.state.time

    def __hand_throw_norm(self):
        palm_forward = np.array([1., 0., 0.])
        mju_rotVecQuat(palm_forward, palm_forward, self.__hand.xquat)
        return palm_forward

    def __hand_throw_ball(self):
        throw_norm = self.__hand_throw_norm()
        self.__set_constraint_active(Constraints.HAND_BALL, False)
        self.ball.xfrc_applied[:3] = self.__hand_throw_strength() * throw_norm
        self.__throw_timer = None

        self.overlay.throw_data = None

    def __hand_throw_strength(self):
        return 100000 * min(2, self.state.time - self.__throw_timer)

    # -----
    # - Interactions: ball & robot
    # --

    def __robot_ball_collisions(self):
        robot_prefix = self.robot.name.split("_")[0]
        ball_prefix = self.ball.name.split("_")[0]
        for c in self.state.data.contact:
            geoms = [self.state.model.geom(g) for g in c.geom]
            names = [g.name for g in geoms]
            if robot_prefix in names[0] and ball_prefix in names[1]:
                return c, geoms[0], geoms[1]
            elif robot_prefix in names[1] and ball_prefix in names[0]:
                return c, geoms[1], geoms[0]
        return None

    def __robot_grab_ball(self, collision, geom_robot, geom_ball):
        cid = self.__constraints[Constraints.ROBOT_BALL]
        assert geom_ball.bodyid == self.ball.id

        # self.state.model.eq_obj1id[cid] = geom_robot.bodyid  # actual colliding body
        # self.state.model.eq_obj1id[cid] = self.__core.bodyid
        # self.__set_connect_anchor(cid, collision.pos)
        # self.__set_constraint_active(Constraints.ROBOT_BALL, True)
        #
        # self.brain.set_has_ball(True)

    # set anchor of connect constraint i to global position pos
    # from https://github.com/google-deepmind/mujoco/issues/229#issuecomment-1176727032
    def __set_connect_anchor(self, i: int, pos: ndarray):
        _, model, data = self.state.unpacked
        if model.eq_type[i] != mjtEq.mjEQ_CONNECT:
            raise TypeError("equality must be a 'connect' for mj_setConnectAnchor")

        # data[0-2] = anchor position in body1 local frame
        id1 = model.eq_obj1id[i]
        # mju_mulMatTVec3(model.eq_data[i][:3], data.xmat[id1], pos - data.xpos[id1])  # actual collision point
        # mju_mulMatTVec3(model.eq_data[i][:3], data.xmat[id1], [0, 0, self.__core.size[2]])  # Top of the head (physically improbable)
        mju_mulMatTVec3(model.eq_data[i][:3], data.xmat[id1], [np.sqrt(2*(self.__core.size[0]**2)), 0, 0])  # Front corner

        # data[3-5] = anchor position in body2 local frame
        id2 = model.eq_obj2id[i]
        mju_mulMatTVec3(model.eq_data[i][3:6], data.xmat[id2], pos - data.xpos[id2])

    # -----
    # - Interactions: hand
    # --

    def __move_hand(self, mouse_move):
        self.__previous_mouse_pos = self._mouse_pos()

        if self._key_pressed(Keys.CTRL):
            scale = .005
            pos = self.state.data.mocap_pos[self.__mocap]
            pos[1:] += scale * mouse_move

        else:
            quat = self.state.data.mocap_quat[self.__mocap]

            scale = .005
            xquat, yquat = np.zeros(4), np.zeros(4)

            mju_euler2Quat(xquat, [0, 0, scale * mouse_move[0] / np.pi], "xyz")
            mju_mulQuat(quat, quat, xquat)

            mju_euler2Quat(yquat, [0, scale * mouse_move[1] / np.pi, 0], "xyz")
            mju_mulQuat(quat, quat, yquat)

    # -----
    # - GUI
    # --

    def _process_keys(self):
        ball_in_hand = self.__is_constraint_active(Constraints.HAND_BALL)

        if self._key_pressed(Keys.LOCK):
            print("[kgd-debug] lock pressed")
            self.__switch_cursor()

        elif self._mouse_down(Buttons.RIGHT) and not ball_in_hand:
            self.__hand_grab_ball()

        elif self._mouse_down(Buttons.LEFT) and ball_in_hand and self.__throw_timer is None:
            self.__hand_prepare_throw_ball()

        elif not self._mouse_down(Buttons.LEFT) and ball_in_hand and self.__throw_timer is not None:
            self.__hand_throw_ball()

        elif ((mouse_move := self.__previous_mouse_pos - self._mouse_pos()) ** 2).sum() > 1 and not self.__cursor_visible:
            self.__move_hand(mouse_move)
