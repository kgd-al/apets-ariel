import math

import mujoco
from PIL.ImageQt import ImageQt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QDialog, QGridLayout, QLabel
from mujoco import viewer, mjtCamLight

from aapets.common.canonical_bodies import get_all
from aapets.common.evaluator import Evaluator
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer


class BodyPicker(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pick a morphology")

        morphologies = {
            name: self.get_image(fn()) for name, fn in get_all().items()
        }

        n = len(morphologies)
        sqrt_n = max(1, int(math.sqrt(n)))

        layout = QGridLayout()
        for ix, (name, img) in enumerate(morphologies.items()):
            i = ix % sqrt_n
            j = ix // sqrt_n
            label = QLabel(name)
            label.setPixmap(img)
            layout.addWidget(label, i, j)
        self.setLayout(layout)

    @staticmethod
    def get_image(body: CoreModule) -> QPixmap:
        world = SimpleFlatWorld()
        # Evaluator.add_defaults(world.spec)

        body.spec.worldbody.add_camera(
            name="tracking-cam",
            mode=mjtCamLight.mjCAMLIGHT_TRACKCOM,
            # pos=(-2, 0, 1.5),
            # xyaxes=[0, -1, 0, 0.75, 0, 0.75],
            pos=(0, 0, 2),
            xyaxes=[0, -1, 0, 1, 0, 0],
        )

        world.spawn(body.spec, spawn_prefix="apet", correct_collision_with_floor=True)

        world.spec.light("light").castshadow = True

        model = world.spec.compile()
        data = mujoco.MjData(model)

        mujoco.viewer.launch(model, data)

        img = single_frame_renderer(
            model, data, width=200, height=200,
            camera="apet1_tracking-cam",
            save=True, save_path="foo.png"
        )

        return QPixmap.fromImage(ImageQt(img))
