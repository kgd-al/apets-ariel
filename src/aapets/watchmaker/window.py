import itertools
from pathlib import Path

from PyQt6.QtCore import Qt, QRect, QSize
from PyQt6.QtGui import QMovie, QResizeEvent
from PyQt6.QtWidgets import QMainWindow, QGridLayout, QWidget, QLabel, QVBoxLayout, QPushButton, QSizePolicy, QFrame

from aapets.watchmaker.config import WatchmakerConfig


class MainWindow(QMainWindow):
    def __init__(self, config: WatchmakerConfig):
        super().__init__()
        self.config = config

        holder = SquareContentWidget()
        layout = QGridLayout(holder.contents)

        self.setCentralWidget(holder)

        gsr = range(config.grid_size)
        self.cells = [GridCell(config) for _ in range(config.population_size)]

        for i, j in itertools.product(gsr, gsr):
            layout.addWidget(self.cells[j*config.grid_size+i], j, i)

    def on_new_generation(self):
        for cell in self.cells:
            cell.viewer.restart()


class SquareContentWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contents = QFrame(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        l = min(self.width(), self.height())
        center = self.rect().center()

        rect = QRect(0, 0, l, l)
        rect.moveCenter(center)
        self.contents.setGeometry(rect)

    def sizeHint(self): return self.contents.sizeHint()


class GridCell(QPushButton):
    def __init__(self, config: WatchmakerConfig):
        super().__init__()

        layout = QVBoxLayout(self)

        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Preferred)

        self.viewer = GifViewer()
        layout.addWidget(self.viewer)

        self.label = QLabel("Hello")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout.addWidget(self.label)

        self.setMinimumSize(config.video_size, config.video_size + self.label.sizeHint().height())

        # self.setStyleSheet("background-color: rgba(255, 0, 0, 255);")
        # self.viewer.setStyleSheet("background-color: rgba(0, 255, 0, 255);")
        # self.label.setStyleSheet("background-color: rgba(0, 0, 255, 255);")

    def update_fields(self, video: Path, fitness: float):
        self.viewer.set_path(str(video))
        self.label.setText(f"{100*fitness:.2f} cm/s")


class GifViewer(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.movie = QMovie()

        self.setMovie(self.movie)
        self.setScaledContents(True)

    def set_path(self, movie: str):
        self.movie.stop()
        self.movie.setFileName(movie)

    def restart(self):
        self.movie.stop()
        self.movie.start()
