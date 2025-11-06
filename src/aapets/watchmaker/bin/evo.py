import argparse
import os
import time

import mujoco
from PyQt6.QtWidgets import QApplication, QDialog

from aapets.common.canonical_bodies import get_all, get
from aapets.watchmaker.body_picker import BodyPicker
from aapets.watchmaker.config import WatchmakerConfig
from aapets.watchmaker.evaluation import compile_world, make_world
from aapets.watchmaker.watchmaker import Watchmaker
from aapets.watchmaker.window import MainWindow
from ariel.utils.renderers import single_frame_renderer


def main(args):
    # app = QApplication([])
    # movie = QMovie("tmp/watchmaker/test-run/0_0.gif")
    # label = QLabel()
    # label.setMovie(movie)
    # movie.start()
    # label.setMinimumSize(label.sizeHint().expandedTo(QSize(100, 100)))
    # label.show()
    # app.exec()
    # exit(42)

    # model, data = compile_world(make_world(get("spider45").spec, .9))
    # mujoco.viewer.launch(model, data)
    # single_frame_renderer(
    #     model, data, width=960, height=960,
    #     camera="apet1_tracking-cam",
    #     save=True, save_path="tmp/watchmaker/test-run/__test.png"
    # )
    # exit(42)

    if args.data_folder.exists() and any(args.data_folder.glob("*")):
        if not args.overwrite:
            raise RuntimeError(f"Output folder already exists, is not empty and overwriting was not requested")
        for item in args.data_folder.glob("*"):
            os.remove(item)
            print("rm", item)
    args.data_folder.mkdir(parents=True, exist_ok=True)

    if args.seed is None:
        args.seed = int(time.time())

    if not args.cache_folder.exists():
        args.cache_folder.mkdir(parents=True)

    app = QApplication([])

    if True or args.body is None:
        picker = BodyPicker(args)
        if picker.exec() == QDialog.DialogCode.Accepted:
            args.body = picker.get_body()
        else:
            args.body = next(iter(get_all().keys()))

    window = MainWindow(args)
    watchmaker = Watchmaker(window, args)

    watchmaker.reset()
    window.show()

    app.exec()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Main evolution script")
    WatchmakerConfig.populate_argparser(parser)
    # parser.print_help()
    parsed_config = parser.parse_args(namespace=WatchmakerConfig())
    parsed_config.update()

    main(parsed_config)
