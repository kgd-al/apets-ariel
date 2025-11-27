import math

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QDialog, QGridLayout, QLabel, QVBoxLayout, QRadioButton

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.utils.renderers import single_frame_renderer
from .config import WatchmakerConfig
from ..common.canonical_bodies import get_all
from ..common.world_builder import make_world, compile_world


class BodyPicker(QDialog):
    def __init__(self, config: WatchmakerConfig):
        super().__init__()
        self.setWindowTitle("Pick a morphology")

        self.body = None

        morphologies = {
            name: self.get_image(fn(), name, config) for name, fn in get_all().items()
        }

        n = len(morphologies)
        sqrt_n = max(1, int(math.sqrt(n)))

        layout = QGridLayout()
        for ix, (name, img) in enumerate(morphologies.items()):
            i = ix % sqrt_n
            j = ix // sqrt_n
            sublayout = QVBoxLayout()
            sublayout.addWidget(label := QLabel())
            sublayout.addWidget(button := QRadioButton(name))

            label.setPixmap(img)
            button.clicked.connect(
                lambda _, _name=name: self.on_click(_name)
            )

            layout.addLayout(sublayout, i, j)
        self.setLayout(layout)
        self.setFixedSize(layout.sizeHint())

    def on_click(self, name):
        self.body = name
        self.accept()

    def get_body(self): return self.body

    @staticmethod
    def get_image(body: CoreModule, name: str, config: WatchmakerConfig) -> QPixmap:
        path = config.cache_folder.joinpath(f"{name}.png")
        if True or not path.exists():
            state, model, data = compile_world(make_world(body.spec, camera_zoom=.95, camera_centered=True))

            # mujoco.viewer.launch(model, data)

            single_frame_renderer(
                model, data, width=200, height=200,
                camera="apet1_tracking-cam",
                save=True, save_path=path,
                transparent=True
            )

            print("Rendering", name, "to", path)

        return QPixmap(str(path))
