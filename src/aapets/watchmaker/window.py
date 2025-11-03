import itertools

from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QMainWindow, QGridLayout, QWidget, QLabel, QVBoxLayout

from aapets.watchmaker.config import WatchmakerConfig


class MainWindow(QMainWindow):
    def __init__(self, config: WatchmakerConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle("CPG-Watchmaker")

        layout = QGridLayout()
        holder = QWidget()

        holder.setLayout(layout)
        self.setCentralWidget(holder)

        gsr = range(config.grid_size)
        self.cells = [[GridCell() for _ in gsr] for _ in gsr]

        for i, j in itertools.product(gsr, gsr):
            layout.addWidget(self.cells[i][j], i, j)


class GridCell(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        # self.viewer = QVideoWidget()
        # layout.addWidget(self.viewer)

        self.label = QLabel("Hello")
        layout.addWidget(self.label)
