import math

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QDialog, QGridLayout, QLabel, QVBoxLayout, QRadioButton

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.utils.renderers import single_frame_renderer
from .config import WatchmakerConfig
from ..common import showcaser
from ..common.canonical_bodies import get_all
from ..common.world_builder import make_world, compile_world


class BodyPicker(QDialog):
    def __init__(self, config: WatchmakerConfig):
        super().__init__()
        self.setWindowTitle("Pick a morphology")

        self.body = None

        morphologies = showcaser.get_all_images(config.cache_folder)

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
