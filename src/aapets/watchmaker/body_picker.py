import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from PyQt6.QtGui import QMovie, QShowEvent
from PyQt6.QtWidgets import QDialog, QGridLayout, QLabel, QVBoxLayout, QRadioButton, QApplication

from aapets.common.config import EvoConfig
from aapets.common.misc.config_base import IntrospectiveAbstractConfig
from aapets.watchmaker.window import GifViewer
from .config import WatchmakerConfig
from ..misc import showcaser


class BodyPicker(QDialog):
    def __init__(self, cache_folder: Path, dynamic=False):
        super().__init__()
        self.setWindowTitle("Pick a morphology")

        self.body = None

        self.cache_folder = cache_folder
        self.dynamic = dynamic

        self.movies = []

        _morphologies = self.morphologies()
        n = len(_morphologies)
        sqrt_n = max(1, int(math.sqrt(n)))

        layout = QGridLayout()
        for ix, (name, img) in enumerate(_morphologies.items()):
            i = ix % sqrt_n
            j = ix // sqrt_n
            sublayout = QVBoxLayout()

            if dynamic:
                sublayout.addWidget(label := GifViewer())
                label.set_path(img)
                self.movies.append(label)
            else:
                sublayout.addWidget(label := QLabel())
                label.setPixmap(img)

            sublayout.addWidget(button := QRadioButton(name))

            button.clicked.connect(
                lambda _, _name=name: self.on_click(_name)
            )

            layout.addLayout(sublayout, i, j)

        self.setLayout(layout)

        if dynamic:
            self.restart_movies()  # to get proper sizes
        self.setFixedSize(layout.sizeHint())

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        if self.dynamic:
            self.restart_movies()

    def restart_movies(self):
        if not self.dynamic:
            print("Cannot restart movies in static mode")
        else:
            for m in self.movies:
                m.restart()

    def on_click(self, name):
        self.body = name
        self.accept()

    def get_body(self): return self.body

    def morphologies(self):
        if self.dynamic:
            return showcaser.get_all_gifs(self.cache_folder)
        else:
            return showcaser.get_all_images(self.cache_folder)


if __name__ == '__main__':
    @dataclass
    class Arguments(IntrospectiveAbstractConfig):
        dynamic: Annotated[bool, "Whether to show animated versions of the iamges"] = False
        cache_folder: Annotated[Path, "Where to look for/store images"] = EvoConfig.cache_folder

    args = Arguments.parse_command_line_arguments(description="Shows all possible canonical bodies")
    app = QApplication([])
    # app.setQuitOnLastWindowClosed(True)
    picker = BodyPicker(args.cache_folder, dynamic=args.dynamic)
    picker.show()
    exit(app.exec())
