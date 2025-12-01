import itertools
import pprint
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Iterator

from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QMainWindow, QGridLayout, QWidget, QLabel, QVBoxLayout, QPushButton, QSizePolicy, QFrame

from aapets.watchmaker.config import WatchmakerConfig
from aapets.watchmaker.types import Individual


class PopulationGrid:
    def __init__(self, layout: int):
        self.grid_size = 3
        self.layout = layout & 255
        self.str_layout = f"{layout:08b}"
        self.str_layout = self.str_layout[:4] + "1" + self.str_layout[4:]
        self.population_size = self.str_layout.count("1")

    def __repr__(self):
        return "\n".join([
            f"Layout: {self.layout}"
        ] + [
            " " + self.str_layout[i * self.grid_size:(i + 1) * self.grid_size]
            for i in range(self.grid_size)
        ])

    def ix(self, i: int, j: int) -> int:
        assert 0 <= i < self.grid_size
        assert 0 <= j < self.grid_size
        return j * self.grid_size + i

    def empty_cell(self, i: int, j: int) -> bool:
        return self.str_layout[self.ix(i, j)] != "1"

    def all_cells(self) -> List[Tuple[int, int]]:
        return [
            (i, j) for j, i in
            itertools.product(range(self.grid_size), range(self.grid_size))
        ]

    def valid_cells(self) -> Iterator[Tuple[int, int]]:
        for i, j in self.all_cells():
            if not self.empty_cell(i, j):
                yield i, j

    @property
    @lru_cache(maxsize=1)
    def parent_ix(self): return self.ix(*self.parent_ij)

    @property
    @lru_cache(maxsize=1)
    def parent_ij(self): return 1, 1


class WatchmakerWindow(QMainWindow):
    def __init__(self, config: WatchmakerConfig):
        super().__init__()
        self.config = config
        grid_spec = PopulationGrid(config.layout)
        if grid_spec.population_size != config.population_size:
            raise ValueError(
                f"Mismatch between population size and grid size/layout"
                f" ({grid_spec.population_size} != {config.population_size})")

        holder = SquareContentWidget()
        layout = QGridLayout(holder.contents)

        self.setCentralWidget(holder)

        self.cells = [
            GridCell((i, j), config) if not grid_spec.empty_cell(i, j) else None
            for i, j in grid_spec.all_cells()
        ]

        for i, j in grid_spec.valid_cells():
            layout.addWidget(self.cells[grid_spec.ix(i, j)], j, i)

        pix = grid_spec.parent_ix
        self.cells = [
            self.cells[pix]
        ] + list(filter(None, self.cells[:pix]+self.cells[pix+1:]))

    def set_interactive_callback(self, callback):
        for ix, cell in enumerate(self.cells):
            cell.clicked.connect(lambda _, _ix=ix: callback(_ix))

    def update_fields(self, ind: Individual, ix: int, generation: int):
        items = []

        if self.config.debug_show_id:
            items.append(f"[R{ind.id}]")

        if ix == 0 and generation > 0:
            items.append("Parent")
        elif generation == 0:
            items.append(f"Robot {ix + 1}")
        else:
            items.append(f"Child {ix}")

        if self.config.show_xspeed:
            items.append(f"{100 * ind.fitness:.2f} cm/s")

        self.cells[ix].update_fields(video=ind.video, desc=" ".join(items))

    def on_new_generation(self):
        for cell in self.cells:
            if cell is not None:
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
    def __init__(self, name, config: WatchmakerConfig):
        super().__init__()
        self.name = name

        layout = QVBoxLayout(self)

        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Preferred)

        self.viewer = GifViewer()
        layout.addWidget(self.viewer)

        self.label = QLabel(f"Hello")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout.addWidget(self.label)

        self.setMinimumSize(config.video_size, config.video_size + self.label.sizeHint().height())

        # self.setStyleSheet("background-color: rgba(255, 0, 0, 255);")
        # self.viewer.setStyleSheet("background-color: rgba(0, 255, 0, 255);")
        # self.label.setStyleSheet("background-color: rgba(0, 0, 255, 255);")

    def __repr__(self):
        return f"GridCell({self.name=}, text={self.label.text()})"

    def update_fields(self, video: Path, desc: str):
        self.viewer.set_path(str(video))
        self.label.setText(desc)


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
