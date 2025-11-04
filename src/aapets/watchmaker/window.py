import itertools

from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QMainWindow, QGridLayout, QWidget, QLabel, QVBoxLayout, QPushButton

from aapets.watchmaker.config import WatchmakerConfig


class MainWindow(QMainWindow):
    def __init__(self, config: WatchmakerConfig):
        super().__init__()
        self.config = config

        layout = QGridLayout()
        holder = QWidget()

        holder.setLayout(layout)
        self.setCentralWidget(holder)

        gsr = range(config.grid_size)
        self.cells = [GridCell(config) for _ in range(config.population_size)]

        for i, j in itertools.product(gsr, gsr):
            layout.addWidget(self.cells[j*config.grid_size+i], j, i)


class GridCell(QPushButton):
    def __init__(self, config: WatchmakerConfig):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        # self.viewer = QVideoWidget()
        self.viewer = QLabel()
        layout.addWidget(self.viewer)

        self.label = QLabel("Hello")
        layout.addWidget(self.label)

        self.setMinimumSize(config.video_size, config.video_size + self.label.sizeHint().height())
