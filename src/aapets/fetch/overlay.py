import numpy as np
from mujoco import mjv_initGeom, mjtGeom, mjv_connector, mju_rotVecQuat
from mujoco.viewer import Handle

from ..common.mujoco.state import MjState
from aapets.fetch.controllers.fetcher import FetcherCPG
from .types import InteractionMode


class FetchOverlay:
    def __init__(self, brain: FetcherCPG, mode: InteractionMode, flags: int = 0xFF):
        self.brain = brain
        self.geom_id = []
        self.mode = mode
        self.flags = flags

        self.throw_data = None

    def start(self, viewer: Handle, state: MjState):
        scene = viewer.user_scn

        n = 6
        self.geom_id = [scene.ngeom+i for i in range(n)]
        scene.ngeom += n

        mjv_initGeom(scene.geoms[self.geom_id[0]],
                     mjtGeom.mjGEOM_ARROW,
                     np.zeros(3), np.zeros(3), np.zeros(9),
                     [1, 1, 0, 1])

        mjv_initGeom(scene.geoms[self.geom_id[1]],
                     mjtGeom.mjGEOM_ARROW,
                     np.zeros(3), np.zeros(3), np.zeros(9),
                     [0, 1, 1, 1])

        mjv_initGeom(scene.geoms[self.geom_id[2]],
                     mjtGeom.mjGEOM_ARROW2,
                     np.zeros(3), np.zeros(3), np.zeros(9),
                     [1, 1, 1, 1])

        mjv_initGeom(scene.geoms[self.geom_id[3]],
                     mjtGeom.mjGEOM_LABEL,
                     np.zeros(3), np.zeros(3), np.zeros(9),
                     [0, 0, 0, 1])

        mjv_initGeom(scene.geoms[self.geom_id[4]],
                     mjtGeom.mjGEOM_ARROW,
                     np.zeros(3), np.zeros(3), np.zeros(9),
                     [1, 1, 1, 1])

        mjv_initGeom(scene.geoms[self.geom_id[5]],
                     mjtGeom.mjGEOM_ARROW,
                     np.zeros(3), np.zeros(3), np.zeros(9),
                     [1, 1, 1, 0])

    def render(self, viewer: Handle, state: MjState):
        scene = viewer.user_scn

        if (self.flags & 1) != 0:
            mjv_connector(scene.geoms[self.geom_id[0]],
                          mjtGeom.mjGEOM_ARROW, .005,
                          self.brain.body.xpos, self.brain.body.xpos + self.brain.forward)

            mjv_connector(scene.geoms[self.geom_id[1]],
                          mjtGeom.mjGEOM_ARROW, .005,
                          self.brain.body.xpos, self.brain.body.xpos + self.brain.goal)

            if (self.flags & 4) != 0:
                scene.geoms[self.geom_id[2]].label = f"a: {self.brain.angle*180/np.pi:+.2f}"
                if self.brain.is_target_visible:
                    scene.geoms[self.geom_id[2]].rgba = [1, 1, 1, 1]
                else:
                    scene.geoms[self.geom_id[2]].rgba = [1, 0, 0, 1]

                fwd_pos = self.brain.body.xpos + .5 * self.brain.forward
                tgt_pos = self.brain.body.xpos + .5 * self.brain.goal
                offset = .5 * (tgt_pos - fwd_pos)
                mjv_connector(scene.geoms[self.geom_id[2]],
                              mjtGeom.mjGEOM_ARROW2, .0025,
                              fwd_pos + offset, tgt_pos + offset)

        if (self.flags & 2) != 0:
            a, b = self.brain.alpha, self.brain.beta
            h = np.array([0, 0, .1])
            d = b * np.array([np.cos(a), np.sin(a), 0])
            mju_rotVecQuat(d, d, self.brain.body.xquat)
            mjv_connector(scene.geoms[self.geom_id[4]],
                          mjtGeom.mjGEOM_ARROW, .01,
                          self.brain.body.xpos + h, self.brain.body.xpos + h + d)

            if (self.flags & 8) != 0:
                scene.geoms[self.geom_id[4]].label = f"alpha: {a:+5.2f}; beta: {b:+5.2f}"

        if (self.flags & 16) != 0:
            arrow = scene.geoms[self.geom_id[5]]
            if self.throw_data is None:
                arrow.rgba[-1] = 0
            else:
                pos = self.brain.target.xpos
                norm = self.throw_data
                arrow.rgba[-1] = 1
                mjv_connector(
                    arrow,
                    mjtGeom.mjGEOM_ARROW, .005,
                    pos, pos + norm,
                )

    def stop(self, viewer: Handle, state: MjState):
        pass

